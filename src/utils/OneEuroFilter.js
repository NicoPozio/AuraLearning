/**
 * Implementazione del 1 Euro Filter (Casiez, Roussel, Vogel, 2012)
 * Ottimizzato per coordinate 2D (Hit Testing dello sguardo).
 */
export class OneEuroFilter {
    constructor(freq = 60, mincutoff = 1.0, beta = 0.007, dcutoff = 1.0) {
        this.freq = freq;           // Frequenza di campionamento stimata (Hz)
        this.mincutoff = mincutoff; // Frequenza di taglio minima (per eliminare il jitter a basse velocità)
        this.beta = beta;           // Coefficiente di adattamento per alte velocità (minimizza il lag)
        this.dcutoff = dcutoff;     // Frequenza di taglio per il calcolo della derivata

        this.xFilter = new LowPassFilter(this._alpha(this.mincutoff));
        this.dxFilter = new LowPassFilter(this._alpha(this.dcutoff));
        this.yFilter = new LowPassFilter(this._alpha(this.mincutoff));
        this.dyFilter = new LowPassFilter(this._alpha(this.dcutoff));
        
        this.lastTime = -1;
    }

    _alpha(cutoff) {
        const te = 1.0 / this.freq;
        const tau = 1.0 / (2 * Math.PI * cutoff);
        return 1.0 / (1.0 + tau / te);
    }

    /**
     * Filtra una nuova coppia di coordinate (X, Y)
     * @param {number} x Coordinata X grezza
     * @param {number} y Coordinata Y grezza
     * @param {number} timestamp Timestamp in millisecondi
     * @returns {Object} Oggetto con le coordinate filtrate {x, y}
     */
    filter(x, y, timestamp) {
        if (this.lastTime !== -1 && timestamp !== undefined) {
            this.freq = 1000.0 / (timestamp - this.lastTime);
        }
        this.lastTime = timestamp;

        // Filtro asse X
        const dx = this.xFilter.hasLastRawValue() ? (x - this.xFilter.lastRawValue()) * this.freq : 0;
        const edx = this.dxFilter.filter(dx, this._alpha(this.dcutoff));
        const cutoffX = this.mincutoff + this.beta * Math.abs(edx);
        const filteredX = this.xFilter.filter(x, this._alpha(cutoffX));

        // Filtro asse Y
        const dy = this.yFilter.hasLastRawValue() ? (y - this.yFilter.lastRawValue()) * this.freq : 0;
        const edy = this.dyFilter.filter(dy, this._alpha(this.dcutoff));
        const cutoffY = this.mincutoff + this.beta * Math.abs(edy);
        const filteredY = this.yFilter.filter(y, this._alpha(cutoffY));

        return { x: filteredX, y: filteredY };
    }
}

class LowPassFilter {
    constructor(alpha) {
        this.setAlpha(alpha);
        this.y = null;
        this.s = null;
    }
    setAlpha(alpha) {
        if (alpha <= 0.0 || alpha > 1.0) throw new Error("Alpha deve essere tra 0 e 1");
        this.a = alpha;
    }
    filter(value, alpha) {
        if (alpha !== undefined) this.setAlpha(alpha);
        if (this.y === null) {
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