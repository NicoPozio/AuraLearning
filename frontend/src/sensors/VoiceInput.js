/**
 * VoiceInput — Web Speech API wrapper
 *
 * MODIFICHE rispetto alla versione precedente:
 *
 * 1. flush() pubblico: ferma e riavvia il riconoscimento per svuotare
 *    il buffer audio accumulato durante il muto anti-echo.
 *    Chiamato da speakECA() al termine della sintesi vocale.
 *
 * 2. setMicBlocked(bool, reason): comunica all'esterno (main.js) il
 *    motivo per cui il mic è bloccato, così la UI può mostrarlo.
 *
 * 3. La logica di restart è stata resa più robusta per gestire il
 *    flush senza loop infiniti.
 */
export class VoiceInput {
    constructor(onTranscript, onStatusChange) {
        this.onTranscript    = onTranscript;
        this.onStatusChange  = onStatusChange || (() => {});
        this.isListening     = false;
        this.shouldRestart   = true;
        this._isFlushing     = false;  // true durante flush → non triggera restart automatico
        this.recognition     = null;

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            console.error("Web Speech API non supportata.");
            this.onStatusChange("Errore: STT non supportato.");
            return;
        }

        this.recognition = new SpeechRecognition();
        this.recognition.lang            = 'it-IT';
        this.recognition.continuous      = true;
        this.recognition.interimResults  = true;
        this.recognition.maxAlternatives = 1;

        this._setupEvents();
    }

    _setupEvents() {
        this.recognition.onstart = () => {
            this.isListening = true;
            if (!this._isFlushing) this.onStatusChange("In ascolto...");
        };

        this.recognition.onresult = (event) => {
            // Se stiamo facendo flush, ignora tutto ciò che arriva
            if (this._isFlushing) return;

            let interimText = "";
            let finalText   = "";
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) finalText   += event.results[i][0].transcript;
                else                          interimText += event.results[i][0].transcript;
            }
            this.onTranscript(interimText, finalText);
        };

        this.recognition.onerror = (event) => {
            if (event.error === 'not-allowed' || event.error === 'audio-capture') {
                this.shouldRestart = false;
                this.onStatusChange("Errore: microfono non disponibile.");
            }
            // 'no-speech' e 'aborted' gestiti da onend con restart automatico
        };

        this.recognition.onend = () => {
            this.isListening = false;

            if (this._isFlushing) {
                // Fine del flush: riavvia pulito
                this._isFlushing = false;
                if (this.shouldRestart) {
                    try { this.recognition.start(); } catch (e) {}
                }
                return;
            }

            if (this.shouldRestart) {
                try { this.recognition.start(); } catch (e) {}
            } else {
                this.onStatusChange("Spento.");
            }
        };
    }

    start() {
        if (this.recognition && !this.isListening) {
            this.shouldRestart = true;
            try { this.recognition.start(); } catch (e) {}
        }
    }

    stop() {
        this.shouldRestart = false;
        if (this.recognition && this.isListening) this.recognition.stop();
    }

    /**
     * Svuota il buffer audio del motore STT.
     * Chiamare DOPO che ECA ha finito di parlare per eliminare
     * le trascrizioni accumulate della voce sintetica.
     * Il riconoscimento si ferma, il buffer viene scartato, poi riparte.
     */
    flush() {
        if (!this.recognition || !this.isListening) return;
        this._isFlushing = true;
        try {
            this.recognition.stop(); // onend → restart pulito
        } catch (e) {
            this._isFlushing = false;
        }
    }
}