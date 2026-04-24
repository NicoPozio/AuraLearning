export class GazeCalibrator {
    constructor() {
        this.calibrationPoints = [];
        this.regressionModel = null;
        this.normParams = null;
    }

    recordDataPoint(irisX, irisY, screenX, screenY) {
        this.calibrationPoints.push({ irisX, irisY, screenX, screenY });
    }

    // [FIX #1] Aggiunto reset() per permettere una ri-calibrazione pulita.
    // PROBLEMA ORIGINALE: se l'utente premeva di nuovo "Calibra Gaze", i nuovi
    // punti venivano AGGIUNTI a quelli vecchi invece di sostituirli, inquinando
    // il modello con dati di sessioni precedenti (posizione testa diversa, luce diversa).
    // Chiama reset() nel btnCalGaze.onclick in main.js prima di calibrationUI.start().
    reset() {
        this.calibrationPoints = [];
        this.regressionModel = null;
        this.normParams = null;
    }

    calculateModel() {
        const n = this.calibrationPoints.length;
        if (n < 6) return false;

        // [FIX #2 — PERFORMANCE] Algoritmo di Welford per media e std in un solo loop.
        // Stesso fix applicato a FrustrationAnalyzer: da 4 passaggi su n punti a 1 passaggio.
        // Qui è più rilevante perché n può essere 9-16 punti di calibrazione (più del doppio
        // rispetto ai 120 campioni di emozione dove ogni iterazione è leggerissima).
        const statsX = this._welfordStats(this.calibrationPoints, p => p.irisX);
        const statsY = this._welfordStats(this.calibrationPoints, p => p.irisY);

        this.normParams = {
            meanX: statsX.mean, stdX: statsX.std,
            meanY: statsY.mean, stdY: statsY.std
        };

        // Costruzione della matrice di Disegno (X) standardizzata
        const X = this.calibrationPoints.map(p => [
            (p.irisX - statsX.mean) / statsX.std,
            (p.irisY - statsY.mean) / statsY.std,
            1 // Termine noto (Bias)
        ]);

        const Yx = this.calibrationPoints.map(p => p.screenX);
        const Yy = this.calibrationPoints.map(p => p.screenY);

        // Risoluzione dei coefficienti con Ridge Regression
        const coeffsX = this._solveRidge(X, Yx);
        const coeffsY = this._solveRidge(X, Yy);

        // [FIX #3] Guard su _solveRidge: se ritorna null (det ≈ 0, matrice singolare)
        // non impostiamo un modello rotto — meglio nessun modello che uno invalido.
        if (!coeffsX || !coeffsY) {
            console.warn("GazeCalibrator: matrice singolare, calibrazione fallita. Aggiungi più punti di calibrazione.");
            return false;
        }

        this.regressionModel = { coeffsX, coeffsY };
        return true;
    }

    predict(irisX, irisY) {
        if (!this.regressionModel || !this.normParams) return null;

        // Applica la stessa standardizzazione Z-Score ai nuovi dati in ingresso
        const nx = (irisX - this.normParams.meanX) / this.normParams.stdX;
        const ny = (irisY - this.normParams.meanY) / this.normParams.stdY;

        const cx = this.regressionModel.coeffsX;
        const cy = this.regressionModel.coeffsY;

        const screenX = nx * cx[0] + ny * cx[1] + cx[2];
        const screenY = nx * cy[0] + ny * cy[1] + cy[2];

        // [FIX #4] Clamping del risultato ai limiti fisici dello schermo.
        // PROBLEMA ORIGINALE: predict() poteva restituire valori negativi o
        // superiori alle dimensioni dello schermo durante movimenti oculari estremi
        // o quando il modello di regressione estrapolava fuori dal range di calibrazione.
        // Questo causava il gazeDot a sparire fuori dal viewport senza errori visibili.
        return {
            x: Math.max(0, Math.min(screenX, window.innerWidth)),
            y: Math.max(0, Math.min(screenY, window.innerHeight))
        };
    }

    // ─── METODI PRIVATI ──────────────────────────────────────────────────────

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
        const lambda = 0.05; // Fattore di regolarizzazione di Tikhonov
        const XT = this._transpose(X);
        const XTX = this._multiply(XT, X);

        // Aggiunta della penalità alla diagonale per garantire la non-singolarità
        XTX[0][0] += lambda;
        XTX[1][1] += lambda;
        XTX[2][2] += lambda;

        const XTy = this._multiplyVec(XT, y);

        // [FIX #3] _solveSystem3x3 ora ritorna null se il determinante è ~0.
        // Propaghiamo null fino a calculateModel() che gestisce il caso.
        return this._solveSystem3x3(XTX, XTy);
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

    _solveSystem3x3(M, b) {
        const det = (m) =>
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
            m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
            m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

        const d = det(M);

        // [FIX #3 — BUG CRITICO] Guard su determinante zero.
        // PROBLEMA ORIGINALE: se tutti i punti di calibrazione erano collineari
        // (es. l'utente fissava solo punti sulla stessa riga orizzontale),
        // det(M) → 0 e la divisione produceva Infinity/NaN silenzioso,
        // corrompendo coeffsX/coeffsY e rendendo predict() inutilizzabile
        // senza nessun messaggio di errore.
        if (Math.abs(d) < 1e-10) return null;

        const replaceCol = (m, col, v) =>
            m.map((r, i) => r.map((c, j) => j === col ? v[i] : c));

        return [
            det(replaceCol(M, 0, b)) / d,
            det(replaceCol(M, 1, b)) / d,
            det(replaceCol(M, 2, b)) / d
        ];
    }
}