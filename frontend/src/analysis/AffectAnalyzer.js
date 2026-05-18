/**
 * AffectAnalyzer — Rilevatore di microespressioni
 *
 * ARCHITETTURA (semplificata rispetto alla versione precedente):
 *
 * Ogni microespressione ha il proprio stato indipendente:
 *   - soglia (z-score rispetto alla baseline personale)
 *   - durata minima richiesta per "attivarsi" (ms)
 *   - tempo da cui il segnale è sopra soglia
 *
 * Lo studente viene classificato "In difficoltà" se ALMENO UNA microespressione
 * è attiva. Ritorna "Normale" dopo RECOVERY_MS ms con tutte le espressioni
 * sotto soglia — timer esplicito, nessun accumulatore.
 *
 * Microespressioni rilevate (tutte da FaceMetricsExtractor):
 *   1. browFurrow     — aggrottamento fronte (AU4)          → durata ≥ 3s
 *   2. eyeSquint      — occhi socchiusi (EAR basso)         → durata ≥ 2.5s
 *   3. lipPress       — labbro morso / compressione (AU24)  → durata ≥ 2.5s
 *   4. mouthOpen      — bocca spalancata (sorpresa)         → durata ≥ 2s
 *   5. noseWrinkle    — naso arricciato/disgusto (AU9)       → durata ≥ 2s
 *   8. mouthFrown     — bocca contratta verso il basso (AU15)    → durata ≥ 2.5s
 *   6. browRaise      — sopracciglio alzato AU1 (perplessità)→ durata ≥ 2.5s
 *   7. faceAbsent     — faccia non rilevata / coperta       → durata ≥ 1.5s
 *
 * Gaze-away (frequenza, non durata — gestito dall'esterno tramite notifyGazeAway):
 *   Se lo sguardo lascia il PDF per GAZE_AWAY_FREQ_COUNT volte
 *   in GAZE_AWAY_FREQ_WINDOW_MS → attiva.
 *
 * Baseline: 120 frame (~4s) di espressione neutra, salvata in sessionStorage.
 * Aggiornamento adattivo lentissimo (alpha 0.002/s) solo durante neutralità certa.
 */
export class AffectAnalyzer {
    constructor(loggerCallback) {
        this.log = loggerCallback || (() => {});

        this.isCalibrating = false;
        this.isCalibrated  = false;
        this.samples  = [];
        this.baseline = this._defaultBaseline();

        // ── Microespressioni ─────────────────────────────────────────────
        // Ogni entry: { zThresh, absThresh, durationMs, _sinceMs, active }
        // zThresh: soglia z-score (relativa alla baseline personale)
        // absThresh: soglia assoluta opzionale (fallback senza calibrazione)
        // durationMs: durata minima per attivazione
        // Ogni entry:
        //   durationMs   = ms sopra soglia per ATTIVARE  (isteresi lenta)
        //   deactivateMs = ms sotto soglia per DISATTIVARE (isteresi rapida)
        //   _sinceMs     = timer attivazione (segnale ON)
        //   _offSinceMs  = timer disattivazione (segnale OFF dopo attivazione)
        this._MEs = {
            browFurrow:  { zThresh: 1.0, durationMs: 1800, deactivateMs: 1200, _sinceMs: 0, _offSinceMs: 0, active: false },
            eyeSquint:   { zThresh: 1.1, durationMs: 1400, deactivateMs: 1000, _sinceMs: 0, _offSinceMs: 0, active: false },
            lipPress:    { zThresh: 1.2, durationMs: 1500, deactivateMs: 1000, _sinceMs: 0, _offSinceMs: 0, active: false },
            mouthOpen:   { zThresh: 1.5, durationMs: 1500, deactivateMs:  800, _sinceMs: 0, _offSinceMs: 0, active: false },
            mouthFrown:  { zThresh: 1.0, durationMs: 1500, deactivateMs: 1200, _sinceMs: 0, _offSinceMs: 0, active: false },
            noseWrinkle: { zThresh: 1.0, durationMs: 1500, deactivateMs:  800, _sinceMs: 0, _offSinceMs: 0, active: false },
            browRaise:   { zThresh: 1.0, durationMs: 1500, deactivateMs: 1000, _sinceMs: 0, _offSinceMs: 0, active: false },
            faceAbsent:  { zThresh: 0,   durationMs: 1500, deactivateMs:  800, _sinceMs: 0, _offSinceMs: 0, active: false },
        };

        // Sorriso: azzera le ME attive (momento di comprensione/sollievo)
        this._SMILE_RESET_THRESH = 2.0; // z-score sorriso


        // ── Recovery ─────────────────────────────────────────────────────
        // Ritorna Normale solo dopo N ms con tutte le ME sotto soglia
        this._RECOVERY_MS  = 4000;
        this._recoverySince = 0; // timestamp da cui siamo in zona neutra

        // ── Stato globale ─────────────────────────────────────────────────
        this.isInDifficulty = false;
        this.activeExpressions = []; // lista nomi ME attive (per debug/log)

        // ── Gaze-away (frequenza) ─────────────────────────────────────────
        this._GAZE_AWAY_WINDOW_MS = 30000;
        this._GAZE_AWAY_COUNT     = 4;    // N volte in finestra → attiva
        this._gazeAwayTimestamps  = [];
        this._gazeAwayActive      = false;

        // ── EMA smoothing sulle metriche grezze ──────────────────────────
        this._SMOOTH_ALPHA   = 0.20;
        this.smoothedMetrics = null;

        // ── Blink rate (per log/debug) ────────────────────────────────────
        this._BLINK_THRESH  = 0.18;
        this._inBlink       = false;
        this.blinkTimestamps = [];
        this.blinkRate      = 0;
    }

    _defaultBaseline() {
        return {
            corrugator:     { mean: 0, std: 1 },
            ear:            { mean: 0, std: 1 },
            lipPress:       { mean: 0, std: 1 },
            mouthOpen:      { mean: 0, std: 1 },
            mouthCurvature: { mean: 0, std: 1 },
            noseWrinkle:    { mean: 0, std: 1 },
            innerBrowRaise: { mean: 0, std: 1 },
            smileIntensity: { mean: 0, std: 1 },
            iod: 0.20
        };
    }

    // ── Calibrazione ──────────────────────────────────────────────────────

    startCalibration() {
        this.isCalibrating  = true;
        this.isCalibrated   = false;
        this.samples        = [];
        this.smoothedMetrics = null;
        this.isInDifficulty = false;
        this.activeExpressions = [];
        this._recoverySince = 0;
        this._gazeAwayTimestamps = [];
        this._gazeAwayActive = false;
        for (const me of Object.values(this._MEs)) { me._sinceMs = 0; me._offSinceMs = 0; me.active = false; }
    }

    processCalibrationSample(metrics) {
        if (!this.isCalibrating) return false;

        this.samples.push(metrics);

        if (this.samples.length >= 120) {
            this.baseline.iod = this.samples.reduce((s, m) => s + m.iod, 0) / this.samples.length;

            const stat = (key, minStd) => {
                const vals    = this.samples.map(m => m[key]).sort((a, b) => a - b);
                const q1      = vals[Math.floor(vals.length * 0.25)];
                const q3      = vals[Math.floor(vals.length * 0.75)];
                const iqr     = q3 - q1;
                const filtered = vals.filter(v => v >= q1 - 1.5 * iqr && v <= q3 + 1.5 * iqr);
                const n        = filtered.length || 1;
                const mean     = filtered.reduce((s, v) => s + v, 0) / n;
                const std      = Math.sqrt(filtered.reduce((s, v) => s + (v - mean) ** 2, 0) / n);
                return { mean, std: Math.max(std, minStd) };
            };

            this.baseline.corrugator     = stat('corrugator',     0.008);
            this.baseline.ear            = stat('ear',            0.020);
            this.baseline.lipPress       = stat('lipPress',       0.010);
            this.baseline.mouthOpen      = stat('mouthOpen',      0.015);
            this.baseline.mouthCurvature = stat('mouthCurvature', 0.010);
            this.baseline.noseWrinkle    = stat('noseWrinkle',    0.008);
            this.baseline.innerBrowRaise = stat('innerBrowRaise', 0.010);
            this.baseline.smileIntensity = stat('smileIntensity', 0.010);

            this.isCalibrating = false;
            this.isCalibrated  = true;
            // Salvataggio baseline in sessionStorage (persiste al refresh)
            try {
                sessionStorage.setItem('aura_baseline', JSON.stringify(this.baseline));
                this.log('Baseline salvata in sessionStorage.', 'INFO');
            } catch (e) {
                this.log('sessionStorage non disponibile.', 'WARN');
            }

            return true;
        }
        return false;
    }

    // Progresso calibrazione (frame validi / richiesti) — per progress bar in main.js
    get calibrationProgress() {
        return Math.min(this._calValidCount / this._CAL_MIN_SAMPLES, 1.0);
    }

    // Tenta di ricaricare la baseline da una sessione precedente
    loadBaselineFromStorage() {
        try {
            const saved = sessionStorage.getItem('aura_baseline');
            if (!saved) return false;
            const parsed = JSON.parse(saved);
            // Merge con i default: chiavi mancanti (vecchie versioni) vengono
            // inizializzate a { mean:0, std:1 } invece di rimanere undefined.
            this.baseline = { ...this._defaultBaseline(), ...parsed };
            this.isCalibrated = true;
            this.log('Baseline caricata da sessionStorage.', 'INFO');
            return true;
        } catch (e) {
            return false;
        }
    }

    // ── Notifica gaze-away dall'esterno (main.js) ─────────────────────────
    notifyGazeAway() {
        const now = performance.now();
        this._gazeAwayTimestamps.push(now);
        // Mantieni solo eventi nella finestra
        const cutoff = now - this._GAZE_AWAY_WINDOW_MS;
        this._gazeAwayTimestamps = this._gazeAwayTimestamps.filter(t => t > cutoff);
        const wasActive = this._gazeAwayActive;
        this._gazeAwayActive = this._gazeAwayTimestamps.length >= this._GAZE_AWAY_COUNT;
        if (this._gazeAwayActive && !wasActive) {
            this.log(`Gaze-away: ${this._gazeAwayTimestamps.length} volte in ${(this._GAZE_AWAY_WINDOW_MS/1000)}s`, 'ALERT');
        }
    }

    // ── Update ────────────────────────────────────────────────────────────

    update(metrics, dtSec) {
        if (!this.isCalibrated) return null;

        const nowMs = performance.now();

        // EMA smoothing
        if (!this.smoothedMetrics) {
            this.smoothedMetrics = { ...metrics };
        } else {
            const a = this._SMOOTH_ALPHA;
            for (const k of ['corrugator','ear','lipPress','mouthOpen','mouthCurvature','noseWrinkle','innerBrowRaise','smileIntensity']) {
                if (metrics[k] !== undefined)
                    this.smoothedMetrics[k] = a * metrics[k] + (1 - a) * this.smoothedMetrics[k];
            }
        }
        const sm = this.smoothedMetrics;

        // Blink rate
        const eyeClosed = sm.ear < this._BLINK_THRESH;
        if (eyeClosed && !this._inBlink) {
            this._inBlink = true;
            this.blinkTimestamps.push(nowMs);
        } else if (!eyeClosed) { this._inBlink = false; }
        this.blinkTimestamps = this.blinkTimestamps.filter(t => t > nowMs - 10000);
        this.blinkRate = this.blinkTimestamps.length / 10;

        // Z-scores rispetto alla baseline personale
        // Guard: se una chiave manca dalla baseline (mismatch versioni), ritorna 0
        const z = (key, smKey) => {
            const b = this.baseline[key];
            if (!b) return 0;
            const val = sm[smKey ?? key];
            if (val === undefined) return 0;
            return (val - b.mean) / (b.std || 1);
        };
        const zCorrugator  = -z('corrugator');       // negato: baseline.corrugator > sm = aggrottato
        const zEar         = -z('ear');              // negato: baseline.ear > sm = socchiuso
        const zLipPress    = -z('lipPress');         // negato: baseline.lipPress > sm = compresso
        const zMouthOpen   =  z('mouthOpen');        // positivo: bocca più aperta della baseline
        const zNoseWrinkle =  z('noseWrinkle');      // positivo: più arricciato
        const zBrowRaise   =  z('innerBrowRaise');   // positivo: sopracciglio alzato
        const zSmile          =  z('smileIntensity');
        const zMouthCurvature =  z('mouthCurvature');  // positivo = commissure basse = frown

        // Mappa z-score → microespressioni
        const signals = {
            browFurrow:  zCorrugator  > this._MEs.browFurrow.zThresh,
            eyeSquint:   zEar         > this._MEs.eyeSquint.zThresh,
            lipPress:    zLipPress    > this._MEs.lipPress.zThresh,
            mouthOpen:   zMouthOpen   > this._MEs.mouthOpen.zThresh,
            noseWrinkle: zNoseWrinkle    > this._MEs.noseWrinkle.zThresh,
            browRaise:   zBrowRaise      > this._MEs.browRaise.zThresh,
            mouthFrown:  zMouthCurvature > this._MEs.mouthFrown.zThresh,
            faceAbsent:  false, // aggiornato da updateFaceAbsent()
        };

        // Sorriso → reset di tutte le ME attive (momento di comprensione)
        if (zSmile > this._SMILE_RESET_THRESH) {
            for (const me of Object.values(this._MEs)) { me._sinceMs = 0; me._offSinceMs = 0; me.active = false; }
            this._gazeAwayActive = false;
            this.log('Sorriso rilevato: reset microespressioni.', 'INFO');
        }

        // Aggiorna stato di ogni ME con logica durata
        const dtMs = dtSec * 1000;
        const newlyActivated = [];

        for (const [name, me] of Object.entries(this._MEs)) {
            if (name === 'faceAbsent') continue; // gestito separatamente

            if (signals[name]) {
                // Segnale ON: accumula timer attivazione, azzera timer disattivazione
                me._sinceMs    += dtMs;
                me._offSinceMs  = 0;
                if (!me.active && me._sinceMs >= me.durationMs) {
                    me.active = true;
                    newlyActivated.push(name);
                    this.log(`ME attivata: ${name} (${(me._sinceMs/1000).toFixed(1)}s)`, 'ALERT');
                }
            } else {
                // Segnale OFF: azzera timer attivazione
                me._sinceMs = 0;
                if (me.active) {
                    // Era attiva: accumula timer disattivazione
                    me._offSinceMs += dtMs;
                    if (me._offSinceMs >= me.deactivateMs) {
                        me.active      = false;
                        me._offSinceMs = 0;
                        this.log(`ME disattivata: ${name}`, 'INFO');
                    }
                }
            }
        }

        // Lista ME attive
        this.activeExpressions = Object.entries(this._MEs)
            .filter(([, me]) => me.active)
            .map(([name]) => name);
        if (this._gazeAwayActive) this.activeExpressions.push('gazeAway');

        const anyActive = this.activeExpressions.length > 0; // include mouthFrown e gazeAway

        // Logica recovery: timer esplicito
        if (!anyActive) {
            if (this._recoverySince === 0) this._recoverySince = nowMs;
            const recoveredMs = nowMs - this._recoverySince;
            if (recoveredMs >= this._RECOVERY_MS && this.isInDifficulty) {
                this.isInDifficulty = false;
                this.log('Ritorno alla neutralità confermato.', 'INFO');
            }
        } else {
            this._recoverySince = 0; // reset recovery se qualcosa è ancora attivo
            if (!this.isInDifficulty) {
                this.isInDifficulty = true;
                this.log(`In difficoltà: [${this.activeExpressions.join(', ')}]`, 'ALERT');
            }
        }

        // Aggiornamento baseline adattivo (lentissimo, solo durante neutralità certa)
        const isNeutralCertain = !anyActive && !this.isInDifficulty && this._recoverySince > 0
            && (nowMs - this._recoverySince) > this._RECOVERY_MS + 2000;

        if (isNeutralCertain) {
            const alpha = 0.002 * dtSec;
            for (const [key, smKey] of [
                ['corrugator','corrugator'], ['ear','ear'], ['lipPress','lipPress'],
                ['mouthOpen','mouthOpen'], ['mouthCurvature','mouthCurvature'], ['noseWrinkle','noseWrinkle'], ['innerBrowRaise','innerBrowRaise']
            ]) {
                const val = sm[smKey];
                if (val !== undefined) {
                    this.baseline[key].mean = (1 - alpha) * this.baseline[key].mean + alpha * val;
                }
            }
        }

        return {
            isInDifficulty: this.isInDifficulty,
            activeExpressions: this.activeExpressions,
            // Metriche grezze per CSV / debug
            zCorrugator,
            zEar,
            zLipPress,
            zMouthOpen,
            zNoseWrinkle,
            zBrowRaise,
            blinkRate: this.blinkRate,
            gazeAwayCount: this._gazeAwayTimestamps.length,
            zMouthCurvature,
            // Segnali grezzi sopra soglia (per debug sidebar)
            debugSignals: {
                browFurrow:  zCorrugator    > this._MEs.browFurrow.zThresh,
                eyeSquint:   zEar           > this._MEs.eyeSquint.zThresh,
                lipPress:    zLipPress      > this._MEs.lipPress.zThresh,
                mouthOpen:   zMouthOpen     > this._MEs.mouthOpen.zThresh,
                mouthFrown:  zMouthCurvature > this._MEs.mouthFrown.zThresh,
                noseWrinkle: zNoseWrinkle   > this._MEs.noseWrinkle.zThresh,
                browRaise:   zBrowRaise     > this._MEs.browRaise.zThresh,
            },
            // z-scores grezzi per calibrazione/diagnostica
            rawZ: { zCorrugator, zEar, zLipPress, zMouthOpen, zMouthCurvature, zNoseWrinkle, zBrowRaise },
        };
    }

    // Chiamato da main.js quando MediaPipe non rileva landmarks (faccia assente)
    updateFaceAbsent(isAbsent, dtSec) {
        const me  = this._MEs.faceAbsent;
        const dtMs = dtSec * 1000;
        if (isAbsent) {
            me._sinceMs    += dtMs;
            me._offSinceMs  = 0;
            if (!me.active && me._sinceMs >= me.durationMs) {
                me.active = true;
                this.log('Faccia non rilevata (mani sul viso?)', 'ALERT');
            }
        } else {
            me._sinceMs = 0;
            if (me.active) {
                me._offSinceMs += dtMs;
                if (me._offSinceMs >= me.deactivateMs) {
                    me.active      = false;
                    me._offSinceMs = 0;
                }
            }
        }
    }
}
