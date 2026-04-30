export class VoiceInput {
    constructor(onTranscript, onStatusChange) {
        this.onTranscript = onTranscript;
        this.onStatusChange = onStatusChange || (() => { });

        this.isListening = false;
        this.shouldRestart = true;
        this.recognition = null;

        // Verifica il supporto cross-browser
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

        if (!SpeechRecognition) {
            console.error("Web Speech API non supportata da questo browser.");
            this.onStatusChange("Errore: STT non supportato dal browser.");
            return;
        }

        this.recognition = new SpeechRecognition();
        this.recognition.lang = 'it-IT'; // Impostato in Italiano
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.maxAlternatives = 1;

        this._setupEvents();
    }

    _setupEvents() {
        this.recognition.onstart = () => {
            this.isListening = true;
            this.onStatusChange("In ascolto...");
        };

        this.recognition.onresult = (event) => {
            let interimText = "";
            let finalText = "";

            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    finalText += event.results[i][0].transcript;
                } else {
                    interimText += event.results[i][0].transcript;
                }
            }

            this.onTranscript(interimText, finalText);
        };

        this.recognition.onerror = (event) => {
            console.warn("[VoiceInput] Errore rilevato:", event.error);
            // Se l'errore è 'no-speech' (silenzio prolungato), la Web Speech API 
            // si ferma. L'evento onend la farà ripartire grazie a shouldRestart.
            if (event.error === 'not-allowed' || event.error === 'audio-capture') {
                this.shouldRestart = false;
                this.onStatusChange("Errore: Accesso al microfono negato.");
            }
        };

        this.recognition.onend = () => {
            this.isListening = false;
            // Riavvio automatico per garantire l'always-listening
            if (this.shouldRestart) {
                try {
                    this.recognition.start();
                } catch (e) {
                    console.error("[VoiceInput] Errore nel riavvio:", e);
                }
            } else {
                this.onStatusChange("Spento.");
            }
        };
    }

    start() {
        if (this.recognition && !this.isListening) {
            this.shouldRestart = true;
            try {
                this.recognition.start();
            } catch (e) {
                // Previene crash se chiamato mentre è già in avvio
            }
        }
    }

    stop() {
        this.shouldRestart = false;
        if (this.recognition && this.isListening) {
            this.recognition.stop();
        }
    }
}