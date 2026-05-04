export class GazeCalibrator {
    constructor() {
        this.calibrationPoints = [];
        this.regressionModel = null;
        this.normParams = null;
    }

    recordDataPoint(irisX, irisY, screenX, screenY) {
        this.calibrationPoints.push({ irisX, irisY, screenX, screenY });
    }

    reset() {
        this.calibrationPoints = [];
        this.regressionModel = null;
        this.normParams = null;
        sessionStorage.removeItem('aura_gaze_model');
    }

    calculateModel() {
        const n = this.calibrationPoints.length;
        // Serve un minimo di 6 punti per risolvere una matrice a 6 incognite
        if (n < 6) {
            console.warn("GazeCalibrator: Troppi pochi punti per la regressione polinomiale (minimo 6).");
            return false;
        }

        const statsX = this._welfordStats(this.calibrationPoints, p => p.irisX);
        const statsY = this._welfordStats(this.calibrationPoints, p => p.irisY);

        this.normParams = {
            meanX: statsX.mean, stdX: statsX.std,
            meanY: statsY.mean, stdY: statsY.std
        };

        // Matrice di design per Polinomiale di 2° grado
        const X = this.calibrationPoints.map(p => {
            const nx = (p.irisX - statsX.mean) / statsX.std;
            const ny = (p.irisY - statsY.mean) / statsY.std;
            return [
                nx,         // x
                ny,         // y
                nx * nx,    // x^2
                ny * ny,    // y^2
                nx * ny,    // xy
                1           // bias
            ];
        });

        const Yx = this.calibrationPoints.map(p => p.screenX);
        const Yy = this.calibrationPoints.map(p => p.screenY);

        const coeffsX = this._solveRidge(X, Yx);
        const coeffsY = this._solveRidge(X, Yy);

        if (!coeffsX || !coeffsY) {
            console.warn("GazeCalibrator: Matrice singolare, calibrazione fallita.");
            return false;
        }

        this.regressionModel = { coeffsX, coeffsY };

        sessionStorage.setItem('aura_gaze_model', JSON.stringify({
            model: this.regressionModel,
            norm: this.normParams
        }));

        return true;
    }

    predict(irisX, irisY) {
        if (!this.regressionModel || !this.normParams) return null;

        const nx = (irisX - this.normParams.meanX) / this.normParams.stdX;
        const ny = (irisY - this.normParams.meanY) / this.normParams.stdY;

        const cx = this.regressionModel.coeffsX;
        const cy = this.regressionModel.coeffsY;

        // Applicazione dei 6 coefficienti polinomiali
        const screenX = cx[0] * nx + cx[1] * ny + cx[2] * (nx * nx) + cx[3] * (ny * ny) + cx[4] * (nx * ny) + cx[5];
        const screenY = cy[0] * nx + cy[1] * ny + cy[2] * (nx * nx) + cy[3] * (ny * ny) + cy[4] * (nx * ny) + cy[5];

        return {
            x: Math.max(0, Math.min(screenX, window.innerWidth)),
            y: Math.max(0, Math.min(screenY, window.innerHeight))
        };
    }

    _welfordStats(arr, getter) {
        let mean = 0, M2 = 0;
        arr.forEach((item, i) => {
            const val = getter(item);
            const delta = val - mean;
            mean += delta / (i + 1);
            M2 += delta * (val - mean);
        });
        return { mean, std: Math.sqrt(M2 / arr.length) || 1 };
    }

    _solveRidge(X, y) {
        const lambda = 0.05; 
        const XT = this._transpose(X);
        const XTX = this._multiply(XT, X);
        
        for (let i = 0; i < XTX.length; i++) {
            XTX[i][i] += lambda;
        }
        
        const XTy = this._multiplyVec(XT, y);
        
        return this._solveGeneralSystem(XTX, XTy);
    }

    // Eliminazione di Gauss con Partial Pivoting (universale per qualsiasi N)
    _solveGeneralSystem(M, b) {
        const n = M.length;
        let A = M.map((row, i) => [...row, b[i]]); 

        for (let i = 0; i < n; i++) {
            let maxRow = i;
            for (let k = i + 1; k < n; k++) {
                if (Math.abs(A[k][i]) > Math.abs(A[maxRow][i])) maxRow = k;
            }
            
            let temp = A[i];
            A[i] = A[maxRow];
            A[maxRow] = temp;

            if (Math.abs(A[i][i]) < 1e-10) return null; 

            for (let k = i + 1; k < n; k++) {
                let factor = A[k][i] / A[i][i];
                for (let j = i; j <= n; j++) {
                    A[k][j] -= factor * A[i][j];
                }
            }
        }

        let x = new Array(n).fill(0);
        for (let i = n - 1; i >= 0; i--) {
            let sum = 0;
            for (let j = i + 1; j < n; j++) {
                sum += A[i][j] * x[j];
            }
            x[i] = (A[i][n] - sum) / A[i][i];
        }
        return x;
    }

    _transpose(A) {
        return A[0].map((_, c) => A.map(r => r[c]));
    }

    _multiply(A, B) {
        return A.map(row =>
            B[0].map((_, i) =>
                row.reduce((acc, _, j) => acc + row[j] * B[j][i], 0)
            )
        );
    }

    _multiplyVec(A, v) {
        return A.map(row => row.reduce((acc, _, i) => acc + row[i] * v[i], 0));
    }

    loadFromStorage() {
        const savedData = sessionStorage.getItem('aura_gaze_model');
        if (savedData) {
            const parsed = JSON.parse(savedData);
            this.regressionModel = parsed.model;
            this.normParams = parsed.norm;
            return true;
        }
        return false;
    }
}