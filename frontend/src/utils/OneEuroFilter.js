/**
 * 1€ Filter (Casiez, Roussel, Vogel, CHI 2012).
 *
 * An adaptive low-pass filter for noisy 2D signals (here, the on-screen
 * gaze cursor). The cutoff frequency is not fixed: it scales with the
 * instantaneous velocity, so the filter smooths aggressively at rest and
 * adds virtually no lag during fast motion.
 *
 *   f_c = mincutoff + β · |ẋ|
 *
 * where ẋ is itself low-pass-filtered at dcutoff to avoid jitter on the
 * derivative signal.
 */
export class OneEuroFilter {
    constructor(freq = 60, mincutoff = 1.0, beta = 0.007, dcutoff = 1.0) {
        this.freq = freq;            // estimated sampling frequency (Hz)
        this.mincutoff = mincutoff;  // minimum cutoff: jitter floor at low speed
        this.beta = beta;            // speed coefficient: how aggressively the cutoff rises
        this.dcutoff = dcutoff;      // cutoff for the derivative filter

        this.xFilter  = new LowPassFilter(this._alpha(this.mincutoff));
        this.dxFilter = new LowPassFilter(this._alpha(this.dcutoff));
        this.yFilter  = new LowPassFilter(this._alpha(this.mincutoff));
        this.dyFilter = new LowPassFilter(this._alpha(this.dcutoff));

        this.lastTime = -1;
    }

    _alpha(cutoff) {
        const te = 1.0 / this.freq;
        const tau = 1.0 / (2 * Math.PI * cutoff);
        return 1.0 / (1.0 + tau / te);
    }

    /**
     * Filter a new (x, y) sample.
     *
     * Guard against anomalous time deltas. If two calls share a
     * timestamp (duplicate) or the timestamp goes backward (timer reset,
     * tab switch, undefined input), this.freq could become Infinity,
     * -Infinity or NaN: freq=Infinity makes _alpha() return 1.0 (no
     * filtering, raw noise passes through); freq=NaN propagates NaN to
     * every filtered coordinate forever. We update freq only when dt is
     * positive and within a sane band (1–200 ms ≈ 5–1000 fps); outside
     * that, we keep the last good freq.
     *
     * @param {number} x         Raw x coordinate.
     * @param {number} y         Raw y coordinate.
     * @param {number} timestamp Milliseconds (performance.now()).
     * @returns {{x:number,y:number}} Filtered coordinates.
     */
    filter(x, y, timestamp) {
        if (this.lastTime !== -1 && timestamp !== undefined) {
            const dtMs = timestamp - this.lastTime;
            if (dtMs > 1 && dtMs < 200) {
                this.freq = 1000.0 / dtMs;
            }
        }
        this.lastTime = timestamp ?? this.lastTime;

        // X axis
        const dx = this.xFilter.hasLastRawValue()
            ? (x - this.xFilter.lastRawValue()) * this.freq
            : 0;
        const edx = this.dxFilter.filter(dx, this._alpha(this.dcutoff));
        const cutoffX = this.mincutoff + this.beta * Math.abs(edx);
        const filteredX = this.xFilter.filter(x, this._alpha(cutoffX));

        // Y axis
        const dy = this.yFilter.hasLastRawValue()
            ? (y - this.yFilter.lastRawValue()) * this.freq
            : 0;
        const edy = this.dyFilter.filter(dy, this._alpha(this.dcutoff));
        const cutoffY = this.mincutoff + this.beta * Math.abs(edy);
        const filteredY = this.yFilter.filter(y, this._alpha(cutoffY));

        return { x: filteredX, y: filteredY };
    }

    /**
     * Re-initialise all internal state. Called by main.js together with
     * gazeCalibrator.reset() so that re-calibrating the gaze does not
     * leave a stale filter state that would make the cursor "jump" from
     * its old position to the first new sample.
     */
    reset() {
        this.xFilter  = new LowPassFilter(this._alpha(this.mincutoff));
        this.dxFilter = new LowPassFilter(this._alpha(this.dcutoff));
        this.yFilter  = new LowPassFilter(this._alpha(this.mincutoff));
        this.dyFilter = new LowPassFilter(this._alpha(this.dcutoff));
        this.lastTime = -1;
    }
}


class LowPassFilter {
    constructor(alpha) {
        this.setAlpha(alpha);
        this.y = null;
        this.s = null;
    }

    /**
     * Silently clamp alpha into (0, 1]. The original 1€ Filter
     * implementation threw on out-of-range alpha, but inside
     * requestAnimationFrame an uncaught exception silently kills the
     * render loop with no visible error. Clamping keeps the filter in a
     * recoverable state: α → 1 passes the raw signal through (no
     * smoothing) and α → 0 freezes it; neither is correct, but neither
     * crashes the system.
     */
    setAlpha(alpha) {
        this.a = Math.max(1e-6, Math.min(1.0, alpha));
    }

    filter(value, alpha) {
        if (alpha !== undefined) this.setAlpha(alpha);
        if (this.y === null) {
            // First sample: seed the state with the raw value so the
            // cursor does not start from (0, 0) and slew towards it.
            this.s = value;
        } else {
            this.s = this.a * value + (1.0 - this.a) * this.s;
        }
        this.y = value;
        return this.s;
    }

    lastRawValue() { return this.y; }
    hasLastRawValue() { return this.y !== null; }
}