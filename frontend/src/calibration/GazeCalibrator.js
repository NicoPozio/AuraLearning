/**
 * GazeCalibrator — Thin-Plate Spline (TPS) mapping from the iris vector
 * (camera space) to the gaze position (screen pixels).
 *
 * Replaces an earlier 2nd-degree polynomial regression for two reasons:
 *   - exact interpolation: f(xᵢ, yᵢ) ≡ screenᵢ at every calibration anchor;
 *   - no polynomial wobble near the screen corners.
 *
 * TPS formula (2D → 1D, applied independently to X and Y):
 *
 *   f(p) = a₀ + a₁·pₓ + a₂·pᵧ + Σᵢ wᵢ · φ(‖p − pᵢ‖),   φ(r) = r² · ln(r²)
 *
 * The weights w and the linear coefficients a are obtained by solving
 * the (N+3) × (N+3) augmented linear system
 *
 *   [ K   P ] [ w ]   [ targets ]
 *   [ Pᵀ  0 ] [ a ] = [    0    ]
 *
 * predict() also returns a confidence score in [0, 1] that decays with
 * distance from the nearest anchor; main.js uses it as an extrapolation
 * guard and to fade the gaze cursor visually.
 */
export class GazeCalibrator {
    constructor() {
        this.calibrationPoints = [];   // raw (iris, screen) pairs from calibration
        // Boolean flag preserved for compatibility with main.js, which
        // gates the prediction code on !!regressionModel.
        this.regressionModel = null;
        this.normParams = null;        // {meanX, stdX, meanY, stdY} for input standardisation
        this._tpsModel = null;         // {wx, wy, norm} — the fitted spline
        this._lastHighConfPos = null;  // cached last high-confidence prediction
    }

    /**
     * Append one (iris, screen) pair to the calibration set. main.js
     * calls this once per anchor with the median of the SPACE-collected
     * samples.
     */
    recordDataPoint(irisX, irisY, screenX, screenY) {
        this.calibrationPoints.push({ irisX, irisY, screenX, screenY });
    }

    /**
     * Forget every calibration sample and the stored model, and clear
     * the persistent backup in sessionStorage. Called at the start of
     * a new calibration.
     */
    reset() {
        this.calibrationPoints = [];
        this.regressionModel = null;
        this._tpsModel = null;
        this._lastHighConfPos = null;
        this.normParams = null;
        sessionStorage.removeItem('aura_gaze_model');
    }

    /**
     * Fit the TPS to the recorded calibration points and persist the
     * model in sessionStorage so it survives a page refresh.
     *
     * @returns {boolean} false on too few points or on a singular linear
     *        system (collinear anchors), true on success.
     */
    calculateModel() {
        const pts = this.calibrationPoints;
        const N = pts.length;

        // TPS needs at least 4 anchors for the affine + non-linear
        // decomposition to be meaningful in 2D.
        if (N < 4) {
            console.warn('GazeCalibrator TPS: servono almeno 4 punti.');
            return false;
        }

        // Input standardisation. The TPS kernel uses Euclidean distance,
        // so the two input axes must live on the same scale. Welford's
        // one-pass algorithm is numerically stable on closely spaced
        // inputs where the naive two-pass formulation suffers from
        // catastrophic cancellation.
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

        // K matrix (N × N). Kernel evaluated on every pair of normalised
        // anchors. The diagonal carries a small Tikhonov regulariser λ:
        // prevents the matrix from becoming ill-conditioned when two
        // anchors lie very close, and produces a smoother surface in
        // extrapolation regions.
        const LAMBDA = 0.05;
        const K = norm.map((pi, i) =>
            norm.map((pj, j) => {
                if (i === j) return LAMBDA;
                const r2 = (pi.nx - pj.nx) ** 2 + (pi.ny - pj.ny) ** 2;
                // r² · ln(r²) is indeterminate (0 · -∞) when r² ≈ 0.
                return r2 > 1e-12 ? r2 * Math.log(r2) : 0;
            })
        );

        // P matrix (N × 3): columns [1, nx, ny]. Encodes the affine part.
        const P = norm.map(p => [1, p.nx, p.ny]);

        // Assemble the (N+3) × (N+3) augmented system:
        //   [ K   P ]
        //   [ Pᵀ  0 ]
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
        // Bottom-right 3×3 block stays zero.

        // RHS targets, padded with three zeros for the bottom block.
        const bx = [...pts.map(p => p.screenX), 0, 0, 0];
        const by = [...pts.map(p => p.screenY), 0, 0, 0];

        // Two independent solves (one per output axis). A is cloned
        // each time because Gaussian elimination is destructive.
        const wx = this._solveSystem(A.map(r => [...r]), bx);
        const wy = this._solveSystem(A.map(r => [...r]), by);

        // Null pivot ⇒ near-collinear calibration anchors.
        if (!wx || !wy) {
            console.warn('GazeCalibrator TPS: sistema singolare.');
            return false;
        }

        this._tpsModel = { wx, wy, norm };
        this.regressionModel = true;

        sessionStorage.setItem('aura_gaze_model', JSON.stringify({
            tps: this._tpsModel,
            norm: this.normParams,
            pts: this.calibrationPoints
        }));
        return true;
    }

    /**
     * Evaluate the fitted TPS at an iris coordinate, clamp to viewport
     * bounds, and blend with the last high-confidence position when the
     * input falls in an extrapolation region.
     *
     * @param {number} irisX - Iris x (depth-corrected by main.js).
     * @param {number} irisY - Iris y.
     * @returns {{x:number,y:number,confidence:number}|null}
     *        Pixel coordinates and a [0, 1] confidence score, or null
     *        if the model has not been fitted yet.
     */
    predict(irisX, irisY) {
        if (!this._tpsModel) return null;

        const { wx, wy, norm } = this._tpsModel;
        const np = this.normParams;
        const N = norm.length;

        // Standardise the query with the same parameters as training.
        const nx = (irisX - np.meanX) / np.stdX;
        const ny = (irisY - np.meanY) / np.stdY;

        // Global affine term: a₀ + a₁·nx + a₂·ny.
        let sx = wx[N] + wx[N + 1] * nx + wx[N + 2] * ny;
        let sy = wy[N] + wy[N + 1] * nx + wy[N + 2] * ny;

        // Squared distance to the closest anchor, used for confidence.
        let minDist2 = Infinity;

        // Non-linear contribution: Σᵢ wᵢ · φ(‖p − pᵢ‖).
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

        // Confidence decays exponentially with the distance from the
        // nearest anchor. The 0.8² denominator sets the falloff radius.
        // main.js fades the gaze dot in proportion to this score so the
        // user perceives uncertainty visually.
        const confidence = Math.exp(-minDist2 / (2 * 0.8 ** 2));

        // Extrapolation guard. Below CONF_THRESHOLD the new TPS
        // prediction is blended linearly with the last position obtained
        // at high confidence — avoids the cursor snapping randomly into
        // the screen corners without degrading the central region.
        const CONF_THRESHOLD = 0.25;
        const clampedX = Math.max(0, Math.min(sx, window.innerWidth));
        const clampedY = Math.max(0, Math.min(sy, window.innerHeight));

        let finalX = clampedX, finalY = clampedY;
        if (confidence < CONF_THRESHOLD && this._lastHighConfPos) {
            const t = confidence / CONF_THRESHOLD;
            finalX = t * clampedX + (1 - t) * this._lastHighConfPos.x;
            finalY = t * clampedY + (1 - t) * this._lastHighConfPos.y;
        } else if (confidence >= CONF_THRESHOLD) {
            // Refresh the cached anchor whenever the prediction is trustworthy.
            this._lastHighConfPos = { x: clampedX, y: clampedY };
        }
        return { x: finalX, y: finalY, confidence };
    }

    /**
     * Restore a TPS model previously saved by calculateModel(). Returns
     * false (without throwing) when no model is found, when the JSON is
     * malformed, or when the saved blob comes from the legacy polynomial
     * calibrator (which lacks the `tps` field).
     */
    loadFromStorage() {
        const saved = sessionStorage.getItem('aura_gaze_model');
        if (!saved) return false;
        try {
            const parsed = JSON.parse(saved);
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

    /**
     * One-pass mean and standard deviation using Welford's algorithm.
     * Numerically stable and O(N). Returns a floored std of at least 1
     * so the downstream standardisation never divides by zero on
     * degenerate constant input.
     */
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

    /**
     * Solve A·x = b by in-place Gaussian elimination with partial
     * pivoting. Returns null when the matrix is numerically singular,
     * signalling the caller to reject the fit.
     */
    _solveSystem(A, b) {
        const n = A.length;
        // Augment A with b as the last column → eliminate in one pass.
        const Aug = A.map((row, i) => [...row, b[i]]);

        // Forward elimination with partial pivoting. Picking the row
        // with the largest |a[k][i]| improves numerical stability and
        // avoids catastrophic cancellation.
        for (let i = 0; i < n; i++) {
            let maxRow = i;
            for (let k = i + 1; k < n; k++)
                if (Math.abs(Aug[k][i]) > Math.abs(Aug[maxRow][i])) maxRow = k;
            [Aug[i], Aug[maxRow]] = [Aug[maxRow], Aug[i]];
            // Tiny pivot ⇒ matrix is singular (collinear anchors).
            if (Math.abs(Aug[i][i]) < 1e-12) return null;
            for (let k = i + 1; k < n; k++) {
                const f = Aug[k][i] / Aug[i][i];
                for (let j = i; j <= n; j++) Aug[k][j] -= f * Aug[i][j];
            }
        }

        // Back substitution.
        const x = new Array(n).fill(0);
        for (let i = n - 1; i >= 0; i--) {
            let sum = 0;
            for (let j = i + 1; j < n; j++) sum += Aug[i][j] * x[j];
            x[i] = (Aug[i][n] - sum) / Aug[i][i];
        }
        return x;
    }
}