/**
 * VoiceInput — wrapper around the browser's Web Speech API.
 *
 * Adds three capabilities on top of the raw SpeechRecognition object:
 *
 *  1. Public `flush()` — stops and restarts the recogniser to discard the
 *     audio buffer accumulated while the microphone was muted (anti-echo
 *     window). Called by `speakECA()` once the TTS playback ends, so the
 *     ECA's own synthetic voice never gets transcribed back as user input.
 *
 *  2. Status reporting — invokes the `onStatusChange` callback so the rest
 *     of the application (main.js) can surface the current mic state and
 *     the reason for any block in the UI.
 *
 *  3. Hardened auto-restart — the recogniser is restarted automatically
 *     after every `end` event (including the ones triggered by `flush()`)
 *     without risking an infinite restart loop.
 *
 * The recogniser is configured in continuous, interim-results mode (it-IT)
 * so that the consumer receives a live stream of partial transcripts plus
 * a finalised transcript at every utterance boundary.
 */
export class VoiceInput {

    /**
     * Set up the Web Speech recogniser and wire the user-supplied callbacks.
     *
     * @param {(interim: string, final: string) => void} onTranscript
     *        Called on every recogniser result. `final` contains the chunks
     *        marked as final by the engine; `interim` contains everything
     *        that is still in progress. Either argument can be the empty
     *        string.
     * @param {(status: string) => void} [onStatusChange]
     *        Optional callback invoked on relevant state changes (listening,
     *        stopped, error). Defaults to a no-op.
     */
    constructor(onTranscript, onStatusChange) {
        this.onTranscript = onTranscript;
        this.onStatusChange = onStatusChange || (() => { });
        this.isListening = false;        // true between `start` and `end` events
        this.shouldRestart = true;         // auto-restart loop guard (disabled on fatal errors)
        this._isFlushing = false;        // true during a flush() cycle; suppresses status churn and result delivery
        this.recognition = null;

        // Vendor-prefixed in Chromium for a long time; both names must be checked.
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            // Bail out early on unsupported browsers (mainly Firefox at the time of writing).
            console.error("Web Speech API non supportata.");
            this.onStatusChange("Errore: STT non supportato.");
            return;
        }

        this.recognition = new SpeechRecognition();
        this.recognition.lang = 'it-IT';   // Italian recogniser
        this.recognition.continuous = true;      // keep listening across sentences
        this.recognition.interimResults = true;      // emit partial transcripts as the user speaks
        this.recognition.maxAlternatives = 1;         // single best hypothesis is enough downstream

        this._setupEvents();
    }

    /**
     * Bind handlers to the underlying SpeechRecognition events. Encapsulates
     * the whole lifecycle (start / result / error / end) and the auto-restart
     * logic, including the special case of a flush() in progress.
     *
     * @private
     */
    _setupEvents() {
        this.recognition.onstart = () => {
            this.isListening = true;
            // Don't update the status during a flush — the mic is just being
            // rebooted internally and the UI shouldn't blink
            if (!this._isFlushing) this.onStatusChange("In ascolto...");
        };

        this.recognition.onresult = (event) => {
            // During a flush we discard every event: the audio buffer being
            // drained may still contain the ECA's own voice
            if (this._isFlushing) return;

            // Concatenate interim and final fragments separately so the
            // consumer can decide whether to act now or wait for the boundary
            let interimText = "";
            let finalText = "";
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) finalText += event.results[i][0].transcript;
                else interimText += event.results[i][0].transcript;
            }
            this.onTranscript(interimText, finalText);
        };

        this.recognition.onerror = (event) => {
            // Permanent failures: stop trying to restart and surface the error
            if (event.error === 'not-allowed' || event.error === 'audio-capture') {
                this.shouldRestart = false;
                this.onStatusChange("Errore: microfono non disponibile.");
            }
            // Transient errors ('no-speech', 'aborted') intentionally fall
            // through: onend will fire next and handle the auto-restart.
        };

        this.recognition.onend = () => {
            this.isListening = false;

            if (this._isFlushing) {
                // End of a flush cycle: clear the flag and start a fresh
                // recogniser session with an empty buffer
                this._isFlushing = false;
                if (this.shouldRestart) {
                    // start() can throw "InvalidStateError" if the engine is
                    // still finalising — swallow it, the next onend will retry
                    try { this.recognition.start(); } catch (e) { }
                }
                return;
            }

            // Normal end: either auto-restart (continuous listening) or
            // report a clean shutdown to the consumer
            if (this.shouldRestart) {
                try { this.recognition.start(); } catch (e) { }
            } else {
                this.onStatusChange("Spento.");
            }
        };
    }

    /**
     * Start the recogniser. Safe to call when already listening (no-op).
     * Also re-arms the auto-restart loop in case it was disabled by a
     * previous `stop()`.
     */
    start() {
        if (this.recognition && !this.isListening) {
            this.shouldRestart = true;
            // Same defensive try/catch as in onend: the engine may still be
            // in a transitional state from a previous session
            try { this.recognition.start(); } catch (e) { }
        }
    }

    /**
     * Stop the recogniser permanently (no auto-restart). The next `start()`
     * call is needed to resume listening.
     */
    stop() {
        this.shouldRestart = false;
        if (this.recognition && this.isListening) this.recognition.stop();
    }

    /**
     * Drain the recogniser's internal audio buffer.
     *
     * Must be invoked AFTER the ECA finishes a TTS utterance, so any audio
     * that leaked from the speakers into the microphone during synthesis
     * is discarded instead of being transcribed back as a user message.
     *
     * Implementation strategy: stop the recogniser; the `onend` handler
     * detects the `_isFlushing` flag and restarts a fresh session with
     * an empty buffer. `onresult` events fired during this window are
     * dropped.
     */
    flush() {
        if (!this.recognition || !this.isListening) return;
        this._isFlushing = true;
        try {
            this.recognition.stop(); // onend will perform a clean restart
        } catch (e) {
            // If stop() throws synchronously the flush would deadlock —
            // reset the flag so future flushes can still proceed
            this._isFlushing = false;
        }
    }
}