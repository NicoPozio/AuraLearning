/**
 * GazeCalibrator — Thin-Plate Spline (TPS) interpolation
 *
 * Sostituisce la regressione polinomiale di 2° grado con TPS.
 *
 * PERCHÉ TPS:
 *   - Interpolazione esatta: f(xᵢ,yᵢ) = screenᵢ per ogni punto di calibrazione
 *   - Nessun "polynomial wobble" agli angoli dello schermo
 *   - Stessa API pubblica del vecchio GazeCalibrator — drop-in replacement
 *   - Costo: O(N²) fit una volta, O(N) per predizione (~9 punti = istantaneo)
 *
 * FORMULA TPS per mapping 2D → 1D (applicata separatamente a X e Y schermo):
 *   f(p) = a₀ + a₁·px + a₂·py + Σᵢ wᵢ · φ(‖p − pᵢ‖)
 *   dove φ(r) = r² · ln(r²)   [kernel thin-plate spline]
 *
 * Sistema lineare (N+3)×(N+3):
 *   ┌ K   P ┐ ┌ w ┐   ┌ targets ┐
 *   └ Pᵀ  0 ┘ └ a ┘ = └    0    ┘
 *
 * API PUBBLICA identica al vecchio GazeCalibrator — drop-in.
 * NUOVO: predict() restituisce anche { confidence } in [0,1]
 */
export class GazeCalibrator {
    constructor() {
        this.calibrationPoints = [];
        this.regressionModel = null; // mantenuto per compatibilità con guard in main.js
        this.normParams = null;
        this._tpsModel = null;
    }

    recordDataPoint(irisX, irisY, screenX, screenY) {
        this.calibrationPoints.push({ irisX, irisY, screenX, screenY });
    }

    reset() {
        this.calibrationPoints = [];
        this.regressionModel = null;
        this._tpsModel = null;
        this.normParams = null;
        sessionStorage.removeItem('aura_gaze_model');
    }

    // ─── Fit ────────────────────────────────────────────────────────────────

    calculateModel() {
        const pts = this.calibrationPoints;
        const N = pts.length;

        if (N < 4) {
            console.warn('GazeCalibrator TPS: servono almeno 4 punti.');
            return false;
        }

        // Normalizzazione input — le distanze nel kernel TPS devono essere uniformi
        const statsIX = this._welfordStats(pts, p => p.irisX);
        const statsIY = this._welfordStats(pts, p => p.irisY);
        this.normParams = {
            meanX: statsIX.mean, stdX: statsIX.std,
            meanY: statsIY.mean, stdY: statsIY.std
        };

        const norm = pts.map(p => ({
            nx: (p.irisX - statsIX.mean) / statsIX.std,
            ny: (p.irisY - statsIY.mean) / statsIY.std,
        }));

        // Matrice K (N×N) — kernel TPS φ(r²) = r² · ln(r²)
        // LAMBDA sulla diagonale: regularizzazione per stabilità numerica
        const LAMBDA = 1e-4;
        const K = norm.map((pi, i) =>
            norm.map((pj, j) => {
                if (i === j) return LAMBDA;
                const r2 = (pi.nx - pj.nx) ** 2 + (pi.ny - pj.ny) ** 2;
                return r2 > 1e-12 ? r2 * Math.log(r2) : 0;
            })
        );

        // Matrice P (N×3): colonne [1, nx, ny]
        const P = norm.map(p => [1, p.nx, p.ny]);

        // Sistema aumentato (N+3)×(N+3)
        const M = N + 3;
        const A = Array.from({ length: M }, () => new Array(M).fill(0));

        for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) A[i][j] = K[i][j];
            A[i][N]     = P[i][0];
            A[i][N + 1] = P[i][1];
            A[i][N + 2] = P[i][2];
            A[N][i]     = P[i][0];
            A[N + 1][i] = P[i][1];
            A[N + 2][i] = P[i][2];
        }
        // Blocco 3×3 in basso a destra = 0 (già inizializzato)

        const bx = [...pts.map(p => p.screenX), 0, 0, 0];
        const by = [...pts.map(p => p.screenY), 0, 0, 0];

        const wx = this._solveSystem(A.map(r => [...r]), bx);
        const wy = this._solveSystem(A.map(r => [...r]), by);

        if (!wx || !wy) {
            console.warn('GazeCalibrator TPS: sistema singolare.');
            return false;
        }

        this._tpsModel = { wx, wy, norm };
        this.regressionModel = true; // flag di compatibilità per main.js

        sessionStorage.setItem('aura_gaze_model', JSON.stringify({
            tps: this._tpsModel,
            norm: this.normParams,
            pts: this.calibrationPoints
        }));

        return true;
    }

    // ─── Predict ────────────────────────────────────────────────────────────

    predict(irisX, irisY) {
        if (!this._tpsModel) return null;

        const { wx, wy, norm } = this._tpsModel;
        const np = this.normParams;
        const N = norm.length;

        const nx = (irisX - np.meanX) / np.stdX;
        const ny = (irisY - np.meanY) / np.stdY;

        // Termine lineare globale (a₀ + a₁·nx + a₂·ny)
        let sx = wx[N] + wx[N + 1] * nx + wx[N + 2] * ny;
        let sy = wy[N] + wy[N + 1] * nx + wy[N + 2] * ny;

        let minDist2 = Infinity;

        for (let i = 0; i < N; i++) {
            const dx = nx - norm[i].nx;
            const dy = ny - norm[i].ny;
            const r2 = dx * dx + dy * dy;

            if (r2 > 1e-12) {
                const phi = r2 * Math.log(r2);
                sx += wx[i] * phi;
                sy += wy[i] * phi;
            }

            if (r2 < minDist2) minDist2 = r2;
        }

        // Confidence [0,1]: decade esponenzialmente con la distanza dal punto di
        // calibrazione più vicino. Utile per modulare l'opacità del gazeDot
        // (bassa confidence = siamo in zona di estrapolazione).
        const confidence = Math.exp(-minDist2 / (2 * 0.6 ** 2));

        return {
            x: Math.max(0, Math.min(sx, window.innerWidth)),
            y: Math.max(0, Math.min(sy, window.innerHeight)),
            confidence
        };
    }

    // ─── Storage ────────────────────────────────────────────────────────────

    loadFromStorage() {
        const saved = sessionStorage.getItem('aura_gaze_model');
        if (!saved) return false;
        try {
            const parsed = JSON.parse(saved);
            // Rifiuta il vecchio formato polynomial (non ha campo 'tps')
            if (!parsed.tps) {
                console.warn('GazeCalibrator: formato legacy rilevato, ricalibra.');
                return false;
            }
            this._tpsModel = parsed.tps;
            this.normParams = parsed.norm;
            this.calibrationPoints = parsed.pts || [];
            this.regressionModel = true;
            return true;
        } catch {
            return false;
        }
    }

    // ─── Utilità ─────────────────────────────────────────────────────────────

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

    // Eliminazione di Gauss con partial pivoting
    _solveSystem(A, b) {
        const n = A.length;
        const Aug = A.map((row, i) => [...row, b[i]]);

        for (let i = 0; i < n; i++) {
            let maxRow = i;
            for (let k = i + 1; k < n; k++)
                if (Math.abs(Aug[k][i]) > Math.abs(Aug[maxRow][i])) maxRow = k;
            [Aug[i], Aug[maxRow]] = [Aug[maxRow], Aug[i]];
            if (Math.abs(Aug[i][i]) < 1e-12) return null;
            for (let k = i + 1; k < n; k++) {
                const f = Aug[k][i] / Aug[i][i];
                for (let j = i; j <= n; j++) Aug[k][j] -= f * Aug[i][j];
            }
        }

        const x = new Array(n).fill(0);
        for (let i = n - 1; i >= 0; i--) {
            let sum = 0;
            for (let j = i + 1; j < n; j++) sum += Aug[i][j] * x[j];
            x[i] = (Aug[i][n] - sum) / Aug[i][i];
        }
        return x;
    }
}