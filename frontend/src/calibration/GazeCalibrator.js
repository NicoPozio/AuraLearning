/**
 * GazeCalibrator — Thin-Plate Spline (TPS) interpolation between the
 * iris vector (camera space) and the gaze position (screen pixels).
 *
 * Replaces an earlier 2nd-degree polynomial regression. Reasons:
 *   • Exact interpolation — f(xᵢ, yᵢ) ≡ screenᵢ at every calibration anchor
 *   • No "polynomial wobble" near the screen corners
 *   • Drop-in replacement — same public API as the previous calibrator
 *   • Cost: O(N²) one-off fit, O(N) per prediction (~9 anchors → instant)
 *
 * TPS FORMULA (2D → 1D mapping, applied independently to the X and Y
 * screen coordinates):
 *
 *   f(p) = a₀ + a₁·px + a₂·py + Σᵢ wᵢ · φ(‖p − pᵢ‖)
 *   where φ(r) = r² · ln(r²)   [thin-plate kernel]
 *
 * The weights w and the linear coefficients a are obtained by solving
 * the (N+3) × (N+3) augmented linear system:
 *
 *   ┌ K   P ┐ ┌ w ┐   ┌ targets ┐
 *   └ Pᵀ  0 ┘ └ a ┘ = └    0    ┘
 *
 * Public API is identical to the previous polynomial implementation.
 * One new field is added to predict()'s return value: a confidence
 * score in [0, 1] that decays with distance from the nearest anchor.
 */
export class GazeCalibrator {
    constructor() {
        this.calibrationPoints = [];   // raw (iris, screen) pairs collected during calibration
        // Boolean flag preserved for compatibility with main.js, which
        // gates the prediction code on `!!regressionModel`
        this.regressionModel = null;
        this.normParams = null;        // {meanX, stdX, meanY, stdY} for input standardisation
        this._tpsModel = null;         // {wx, wy, norm} — the fitted spline
        this._lastHighConfPos = null;  // cached last high-confidence prediction (extrapolation guard)
    }

    /**
     * Append one (iris, screen) pair to the calibration set. main.js calls
     * this once per anchor with the median of the SPACE-collected samples.
     *
     * @param {number} irisX  - Iris x in normalised camera space (depth-corrected).
     * @param {number} irisY  - Iris y in the same space.
     * @param {number} screenX - Target screen pixel x for this anchor.
     * @param {number} screenY - Target screen pixel y for this anchor.
     */
    recordDataPoint(irisX, irisY, screenX, screenY) {
        this.calibrationPoints.push({ irisX, irisY, screenX, screenY });
    }

    /**
     * Forget every calibration sample and stored model, and clear the
     * persistent backup in sessionStorage. Called at the start of every
     * new calibration session.
     */
    reset() {
        this.calibrationPoints = [];
        this.regressionModel = null;
        this._tpsModel = null;
        this._lastHighConfPos = null;
        this.normParams = null;
        sessionStorage.removeItem('aura_gaze_model');
    }

    // ─── Fit ────────────────────────────────────────────────────────────────

    /**
     * Fit the TPS to the recorded calibration points and persist the
     * model in sessionStorage so it survives a page refresh.
     *
     * Returns false if there are too few points or if the linear system
     * turns out to be singular (collinear anchors).
     *
     * @returns {boolean} true on success, false otherwise.
     */
    calculateModel() {
        const pts = this.calibrationPoints;
        const N = pts.length;

        // TPS needs at least N = 4 anchors to make the affine + non-linear
        // decomposition meaningful in 2D
        if (N < 4) {
            console.warn('GazeCalibrator TPS: servono almeno 4 punti.');
            return false;
        }

        // ── Input standardisation ────────────────────────────────────────
        // The TPS kernel uses Euclidean distance, so the two input axes
        // must live on the same scale. We z-normalise the iris coordinates
        // using Welford's online algorithm (numerically stable mean / std).
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

        // ── Build the K matrix (N × N) ───────────────────────────────────
        // Kernel evaluated on every pair of normalised anchors.
        // The diagonal carries a small regulariser λ (Tikhonov-like): it
        // prevents the matrix from becoming ill-conditioned when two
        // anchors lie very close to each other, and produces a smoother
        // surface in extrapolation regions.
        const LAMBDA = 0.05; // bumped up: stable smoothing on extrapolation
        const K = norm.map((pi, i) =>
            norm.map((pj, j) => {
                if (i === j) return LAMBDA;
                const r2 = (pi.nx - pj.nx) ** 2 + (pi.ny - pj.ny) ** 2;
                // Guard against r² ≈ 0 where r²·ln(r²) is indeterminate (0·-∞)
                return r2 > 1e-12 ? r2 * Math.log(r2) : 0;
            })
        );

        // ── Build the P matrix (N × 3): columns [1, nx, ny] ─────────────
        // Encodes the affine part of the TPS expansion
        const P = norm.map(p => [1, p.nx, p.ny]);

        // ── Assemble the (N+3) × (N+3) augmented system A ───────────────
        // Block structure:  [ K   P  ]
        //                   [ Pᵀ  0  ]
        const M = N + 3;
        const A = Array.from({ length: M }, () => new Array(M).fill(0));

        for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) A[i][j] = K[i][j];
            // Top-right K | P
            A[i][N] = P[i][0];
            A[i][N + 1] = P[i][1];
            A[i][N + 2] = P[i][2];
            // Bottom-left Pᵀ
            A[N][i] = P[i][0];
            A[N + 1][i] = P[i][1];
            A[N + 2][i] = P[i][2];
        }
        // Bottom-right 3×3 block stays zero (already initialised that way)

        // Right-hand side targets, padded with three zeros for the
        // bottom block of the system
        const bx = [...pts.map(p => p.screenX), 0, 0, 0];
        const by = [...pts.map(p => p.screenY), 0, 0, 0];

        // Two independent solves (one per output axis). A is cloned each
        // time because Gaussian elimination is in-place and destructive.
        const wx = this._solveSystem(A.map(r => [...r]), bx);
        const wy = this._solveSystem(A.map(r => [...r]), by);

        // _solveSystem returns null on singular pivot — typical cause is
        // a near-collinear calibration (all anchors on roughly one line)
        if (!wx || !wy) {
            console.warn('GazeCalibrator TPS: sistema singolare.');
            return false;
        }

        this._tpsModel = { wx, wy, norm };
        this.regressionModel = true; // compatibility flag for main.js guards

        // Persist the entire model so a page refresh doesn't lose calibration
        sessionStorage.setItem('aura_gaze_model', JSON.stringify({
            tps: this._tpsModel,
            norm: this.normParams,
            pts: this.calibrationPoints
        }));

        return true;
    }

    // ─── Predict ────────────────────────────────────────────────────────────

    /**
     * Evaluate the fitted TPS at an iris coordinate, clamp the result to
     * the viewport bounds, and blend with the last high-confidence
     * position when the input falls in an extrapolation region.
     *
     * @param {number} irisX - Iris x in camera space (depth-corrected by main.js).
     * @param {number} irisY - Iris y in the same space.
     * @returns {{x:number,y:number,confidence:number}|null}
     *          Pixel coordinates and a [0,1] confidence score, or null if
     *          the model has not been fitted yet.
     */
    predict(irisX, irisY) {
        if (!this._tpsModel) return null;

        const { wx, wy, norm } = this._tpsModel;
        const np = this.normParams;
        const N = norm.length;

        // Standardise the query point with the same parameters as the training set
        const nx = (irisX - np.meanX) / np.stdX;
        const ny = (irisY - np.meanY) / np.stdY;

        // Global affine term: a₀ + a₁·nx + a₂·ny
        let sx = wx[N] + wx[N + 1] * nx + wx[N + 2] * ny;
        let sy = wy[N] + wy[N + 1] * nx + wy[N + 2] * ny;

        // Track the squared distance to the closest anchor — used for the
        // confidence score below
        let minDist2 = Infinity;

        // Non-linear contribution: Σᵢ wᵢ · φ(‖p − pᵢ‖)
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

        // ── Confidence ∈ [0, 1] ──────────────────────────────────────────
        // Decays exponentially with the distance from the nearest anchor.
        // The denominator (2 · 0.8²) sets the falloff radius — a bigger
        // value means we trust the spline farther out. main.js uses this
        // score to modulate the gaze dot's opacity (low confidence = the
        // cursor visually fades, conveying uncertainty to the user).
        const confidence = Math.exp(-minDist2 / (2 * 0.8 ** 2));

        // ── Extrapolation guard ──────────────────────────────────────────
        // When confidence drops below CONF_THRESHOLD we blend the new TPS
        // prediction with the last position obtained at high confidence,
        // linearly weighted by `t = confidence / threshold`. This avoids
        // the cursor snapping randomly into the screen corners without
        // degrading accuracy in the well-calibrated central region.
        const CONF_THRESHOLD = 0.25;
        const clampedX = Math.max(0, Math.min(sx, window.innerWidth));
        const clampedY = Math.max(0, Math.min(sy, window.innerHeight));

        let finalX = clampedX, finalY = clampedY;
        if (confidence < CONF_THRESHOLD && this._lastHighConfPos) {
            // t = 0 → fully use the cached high-confidence point
            // t = 1 → fully use the current TPS prediction
            const t = confidence / CONF_THRESHOLD;
            finalX = t * clampedX + (1 - t) * this._lastHighConfPos.x;
            finalY = t * clampedY + (1 - t) * this._lastHighConfPos.y;
        } else if (confidence >= CONF_THRESHOLD) {
            // Refresh the cached anchor whenever the prediction is trustworthy
            this._lastHighConfPos = { x: clampedX, y: clampedY };
        }

        return { x: finalX, y: finalY, confidence };
    }

    // ─── Storage ────────────────────────────────────────────────────────────

    /**
     * Attempt to restore a TPS model previously saved by calculateModel().
     * Returns false (and does not throw) when no model is found, when the
     * JSON is malformed, or when the saved blob comes from the legacy
     * polynomial calibrator (which lacks the `tps` field).
     *
     * @returns {boolean} true on successful restoration.
     */
    loadFromStorage() {
        const saved = sessionStorage.getItem('aura_gaze_model');
        if (!saved) return false;
        try {
            const parsed = JSON.parse(saved);
            // Reject the legacy polynomial format (no 'tps' field)
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

    // ─── Utilities ───────────────────────────────────────────────────────────

    /**
     * One-pass mean / standard deviation using Welford's algorithm.
     * Numerically stable and O(N), avoids the catastrophic cancellation
     * that affects the naive two-pass formulation on closely spaced inputs.
     *
     * @param {Array<*>} arr     - The input collection.
     * @param {(item:*)=>number} getter - Projector from item to numeric value.
     * @returns {{mean:number, std:number}} Sample mean and std (>= 1 fallback).
     * @private
     */
    _welfordStats(arr, getter) {
        let mean = 0, M2 = 0;
        arr.forEach((item, i) => {
            const val = getter(item);
            const delta = val - mean;
            mean += delta / (i + 1);
            M2 += delta * (val - mean);
        });
        // Floor at 1 to guarantee non-zero divisor for the downstream
        // standardisation step (degenerate constant input would otherwise
        // give std = 0)
        return { mean, std: Math.sqrt(M2 / arr.length) || 1 };
    }

    /**
     * Solve A·x = b by in-place Gaussian elimination with partial pivoting.
     * Returns null when the matrix is (numerically) singular, signalling
     * the caller to reject the fit.
     *
     * @param {number[][]} A - Square coefficient matrix (mutated in place).
     * @param {number[]}  b  - Right-hand side.
     * @returns {number[]|null} Solution vector x, or null on singular A.
     * @private
     */
    _solveSystem(A, b) {
        const n = A.length;
        // Augment A with b as the last column → eliminate in one pass
        const Aug = A.map((row, i) => [...row, b[i]]);

        // ── Forward elimination with partial pivoting ────────────────────
        for (let i = 0; i < n; i++) {
            // Pick the row with the largest |a[k][i]| as the pivot — improves
            // numerical stability and avoids catastrophic cancellation
            let maxRow = i;
            for (let k = i + 1; k < n; k++)
                if (Math.abs(Aug[k][i]) > Math.abs(Aug[maxRow][i])) maxRow = k;
            [Aug[i], Aug[maxRow]] = [Aug[maxRow], Aug[i]];
            // Tiny pivot → matrix is singular (e.g. collinear anchors)
            if (Math.abs(Aug[i][i]) < 1e-12) return null;
            // Subtract the pivot row from every row below to zero out column i
            for (let k = i + 1; k < n; k++) {
                const f = Aug[k][i] / Aug[i][i];
                for (let j = i; j <= n; j++) Aug[k][j] -= f * Aug[i][j];
            }
        }

        // ── Back substitution ────────────────────────────────────────────
        const x = new Array(n).fill(0);
        for (let i = n - 1; i >= 0; i--) {
            let sum = 0;
            for (let j = i + 1; j < n; j++) sum += Aug[i][j] * x[j];
            x[i] = (Aug[i][n] - sum) / Aug[i][i];
        }
        return x;
    }
}