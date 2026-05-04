/**
 * Implementazione del 1 Euro Filter (Casiez, Roussel, Vogel, 2012)
 * Ottimizzato per coordinate 2D (Hit Testing dello sguardo).
 */
export class OneEuroFilter {
    constructor(freq = 60, mincutoff = 1.0, beta = 0.007, dcutoff = 1.0) {
        this.freq = freq;       // Frequenza di campionamento stimata (Hz)
        this.mincutoff = mincutoff;  // Frequenza di taglio minima (jitter a basse velocità)
        this.beta = beta;       // Coefficiente di adattamento per alte velocità (lag)
        this.dcutoff = dcutoff;    // Frequenza di taglio per la derivata

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
     * Filtra una nuova coppia di coordinate (X, Y).
     * @param {number} x         Coordinata X grezza
     * @param {number} y         Coordinata Y grezza
     * @param {number} timestamp Timestamp in millisecondi (performance.now())
     * @returns {{ x: number, y: number }} Coordinate filtrate
     */
    filter(x, y, timestamp) {
        // [FIX #1 — BUG CRITICO] Guard su delta temporale anomalo.
        //
        // PROBLEMA ORIGINALE: se due chiamate a filter() arrivavano con lo stesso
        // timestamp (duplicato) o con un timestamp più piccolo del precedente
        // (es. reset del timer, cambio tab, timestamp undefined), this.freq
        // diventava Infinity, -Infinity o NaN.
        // Un freq = Infinity fa sì che _alpha() restituisca 1.0 → il LowPassFilter
        // non filtra più nulla (passa tutto il rumore grezzo).
        // Un freq = NaN propaga NaN a tutte le coordinate filtrate per sempre.
        //
        // SOLUZIONE: aggiorniamo freq solo se il delta è strettamente positivo
        // e ragionevole (tra 1ms e 200ms = tra 5fps e 1000fps).
        if (this.lastTime !== -1 && timestamp !== undefined) {
            const dtMs = timestamp - this.lastTime;
            if (dtMs > 1 && dtMs < 200) {
                this.freq = 1000.0 / dtMs;
            }
            // Se dtMs è fuori range, teniamo l'ultimo freq valido — il filtro
            // continua a funzionare con la frequenza stimata più recente.
        }
        this.lastTime = timestamp ?? this.lastTime;

        // ─── ASSE X ──────────────────────────────────────────────────────────
        const dx = this.xFilter.hasLastRawValue()
            ? (x - this.xFilter.lastRawValue()) * this.freq
            : 0;
        const edx = this.dxFilter.filter(dx, this._alpha(this.dcutoff));
        const cutoffX = this.mincutoff + this.beta * Math.abs(edx);
        const filteredX = this.xFilter.filter(x, this._alpha(cutoffX));

        // ─── ASSE Y ──────────────────────────────────────────────────────────
        const dy = this.yFilter.hasLastRawValue()
            ? (y - this.yFilter.lastRawValue()) * this.freq
            : 0;
        const edy = this.dyFilter.filter(dy, this._alpha(this.dcutoff));
        const cutoffY = this.mincutoff + this.beta * Math.abs(edy);
        const filteredY = this.yFilter.filter(y, this._alpha(cutoffY));

        return { x: filteredX, y: filteredY };
    }

    // [FIX #2] Aggiunto reset() per coerenza con il pattern del resto del progetto.
    // PROBLEMA ORIGINALE: se l'utente ricalibrava il gaze, il filtro manteneva
    // lo stato interno (y, s) della sessione precedente. Al primo frame dopo la
    // ricalibrazione, il filtro partiva da una posizione schermo completamente
    // diversa e produceva un glitch visivo (il gazeDot "saltava" dalla vecchia
    // posizione alla nuova invece di partire pulito).
    // Chiamare reset() in main.js insieme a gazeCalibrator.reset().
    reset() {
        this.xFilter = new LowPassFilter(this._alpha(this.mincutoff));
        this.dxFilter = new LowPassFilter(this._alpha(this.dcutoff));
        this.yFilter = new LowPassFilter(this._alpha(this.mincutoff));
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

    setAlpha(alpha) {
        // [FIX #3 — BUG SILENZIOSO] La validazione originale lanciava un'eccezione
        // per alpha fuori range, ma non c'era nessun try/catch a valle che la
        // gestisse. Un cutoff molto alto o molto basso (es. causato da freq = Infinity
        // dopo il bug #1) produceva alpha = 0 o alpha > 1, il throw interrompeva
        // il loop di rendering silenziosamente (requestAnimationFrame non propaga
        // eccezioni non catturate in modo visibile) e il gazeDot smetteva di muoversi
        // senza nessun messaggio di errore evidente in produzione.
        //
        // SOLUZIONE: clamp silenzioso invece di throw.
        // Alpha = 1.0  → passa tutto il segnale grezzo (nessun smoothing, ma non si rompe)
        // Alpha → 0.0  → filtra tutto (segnale frozen, ma non si rompe)
        // In entrambi i casi il filtro rimane in uno stato valido e recuperabile.
        this.a = Math.max(1e-6, Math.min(1.0, alpha));
    }

    filter(value, alpha) {
        if (alpha !== undefined) this.setAlpha(alpha);
        if (this.y === null) {
            // Prima chiamata: inizializza lo stato direttamente al valore grezzo
            // (nessun filtro al primo sample — comportamento corretto per evitare
            // lo "startup transient" che farebbe partire il gazeDot da (0,0))
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