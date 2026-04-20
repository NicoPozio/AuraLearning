export class GazeCalibrator {
    constructor() {
        this.calibrationPoints = []; 
        this.regressionModel = null;
        this.normParams = null;
    }

    recordDataPoint(irisX, irisY, screenX, screenY) {
        this.calibrationPoints.push({ irisX, irisY, screenX, screenY });
    }

    calculateModel() {
        const n = this.calibrationPoints.length;
        if (n < 6) return false;

        // 1. Z-Score Normalization: Calcolo Media e Deviazione Standard
        const meanX = this.calibrationPoints.reduce((s, p) => s + p.irisX, 0) / n;
        const meanY = this.calibrationPoints.reduce((s, p) => s + p.irisY, 0) / n;
        
        const stdX = Math.sqrt(this.calibrationPoints.reduce((s, p) => s + Math.pow(p.irisX - meanX, 2), 0) / n) || 1;
        const stdY = Math.sqrt(this.calibrationPoints.reduce((s, p) => s + Math.pow(p.irisY - meanY, 2), 0) / n) || 1;

        this.normParams = { meanX, meanY, stdX, stdY };

        // 2. Costruzione della matrice di Disegno (X) standardizzata
        const X = this.calibrationPoints.map(p => [
            (p.irisX - meanX) / stdX,
            (p.irisY - meanY) / stdY,
            1 // Termine noto (Bias)
        ]);

        const Yx = this.calibrationPoints.map(p => p.screenX);
        const Yy = this.calibrationPoints.map(p => p.screenY);

        // 3. Risoluzione dei coefficienti
        this.regressionModel = {
            coeffsX: this._solveRidge(X, Yx),
            coeffsY: this._solveRidge(X, Yy)
        };
        
        return true;
    }

    predict(irisX, irisY) {
        if (!this.regressionModel) return null;
        
        // Applica la stessa standardizzazione Z-Score ai nuovi dati in ingresso
        const nx = (irisX - this.normParams.meanX) / this.normParams.stdX;
        const ny = (irisY - this.normParams.meanY) / this.normParams.stdY;

        const cx = this.regressionModel.coeffsX;
        const cy = this.regressionModel.coeffsY;

        const screenX = nx * cx[0] + ny * cx[1] + cx[2];
        const screenY = nx * cy[0] + ny * cy[1] + cy[2];

        return { x: screenX, y: screenY };
    }

    _solveRidge(X, y) {
        const lambda = 0.05; // Fattore di regolarizzazione di Tikhonov
        const XT = this._transpose(X);
        const XTX = this._multiply(XT, X);

        // Aggiunta della penalità alla diagonale per garantire la non-singolarità
        XTX[0][0] += lambda;
        XTX[1][1] += lambda;
        XTX[2][2] += lambda;

        const XTy = this._multiplyVec(XT, y);
        return this._solveSystem3x3(XTX, XTy);
    }

    _transpose(A) { return A[0].map((_, c) => A.map(r => r[c])); }
    
    _multiply(A, B) {
        return A.map(row => B[0].map((_, i) => row.reduce((acc, _, j) => acc + row[j] * B[j][i], 0)));
    }
    
    _multiplyVec(A, v) {
        return A.map(row => row.reduce((acc, _, i) => acc + row[i] * v[i], 0));
    }
    
    _solveSystem3x3(M, b) {
        const det = (m) => m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]) - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0]) + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
        const d = det(M);
        
        const replaceCol = (m, col, v) => m.map((r, i) => r.map((c, j) => j === col ? v[i] : c));
        return [
            det(replaceCol(M, 0, b)) / d,
            det(replaceCol(M, 1, b)) / d,
            det(replaceCol(M, 2, b)) / d
        ];
    }
}