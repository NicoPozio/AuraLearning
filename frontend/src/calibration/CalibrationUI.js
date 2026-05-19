/**
 * CalibrationUI — full-screen overlay that walks the user through the
 * 9-point gaze calibration procedure.
 *
 * Responsibility split with main.js:
 *   • This class handles ONLY presentation: showing the dots, drawing the
 *     progress ring, transitioning between the intro screen and the dot
 *     sequence.
 *   • main.js handles the actual sample collection (frame-by-frame iris
 *     readings while SPACE is held) and calls back into updateProgress()
 *     and advance() at the right moments.
 *
 * Lifecycle:
 *   start()              → overlay visible, intro screen
 *   onSpaceDown()        → transitions intro → dots
 *   updateProgress(p)    → animates the ring fill while SPACE is held
 *   advance()            → moves to the next dot (called by main.js when
 *                          enough samples have been collected)
 *   onComplete()         → callback fired after the 9th dot is recorded
 */
export class CalibrationUI {

    /**
     * Define the 9 anchor positions (in [0,1] viewport-relative coords),
     * create the overlay element, and store the completion callback.
     *
     * @param {() => void} onComplete - Invoked once the user has confirmed
     *        all 9 anchors. Typical implementation: trigger TPS fit and
     *        show the gaze dot.
     */
    constructor(onComplete) {
        this.onComplete = onComplete;
        // Anchor layout: 4 corners + centre + 2 edges + 2 lower-band points.
        // The lower-band points (y=0.65) are slightly off-axis on purpose —
        // they help the TPS interpolate the reading area of a PDF, which
        // typically sits in the lower half of the screen.
        this.points = [
            { x: 0.1, y: 0.1 }, { x: 0.9, y: 0.1 },   // top corners
            { x: 0.1, y: 0.9 }, { x: 0.9, y: 0.9 },   // bottom corners
            { x: 0.5, y: 0.5 },                       // centre
            { x: 0.5, y: 0.1 }, { x: 0.5, y: 0.9 },   // top/bottom edge midpoints
            { x: 0.25, y: 0.65 }, { x: 0.75, y: 0.65 }, // lower-band off-axis anchors
        ];
        this.currentIndex = 0;
        this._phase = 'intro'; // 'intro' = welcome screen, 'dots' = active calibration
        this.overlay = this._createOverlay();
        this._ringEl = null;    // SVG <circle> currently being animated (progress ring)
        this._ringPerim = 0;     // cached perimeter for stroke-dashoffset calculations
        this._labelEl = null;    // bottom instruction label (mutated by updateProgress)
    }

    /**
     * Build the full-viewport overlay <div> once and attach it to <body>.
     * Hidden by default; revealed by start().
     *
     * @returns {HTMLDivElement} The overlay element.
     * @private
     */
    _createOverlay() {
        const div = document.createElement('div');
        // cursor:none hides the OS pointer during calibration so it doesn't
        // distract the user from looking at the actual dot
        div.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;background:rgba(255,255,255,0.95);z-index:9999;display:none;cursor:none;font-family:system-ui,sans-serif;';
        document.body.appendChild(div);
        return div;
    }

    /**
     * Show the calibration overlay and reset the state machine to the
     * intro screen. Called by main.js when the user clicks the
     * "Calibrazione Sguardo" button.
     */
    start() {
        this.overlay.style.display = 'block';
        this._phase = 'intro';
        this.currentIndex = 0;
        this._showIntro();
    }

    /**
     * Handle a SPACE keydown from main.js. The first press dismisses the
     * intro and shows the first dot; subsequent presses are consumed by
     * main.js itself for sample collection — this method only cares
     * about the intro → dots transition.
     */
    onSpaceDown() {
        if (this._phase === 'intro') {
            this._phase = 'dots';
            this._showNextPoint();
        }
        // Once in 'dots' phase, multi-frame sample collection is owned by main.js
    }

    /**
     * Render the welcome screen: an icon, a short explanation of what is
     * about to happen, and a hint that SPACE starts the procedure.
     *
     * @private
     */
    _showIntro() {
        this.overlay.innerHTML = '';

        const box = document.createElement('div');
        box.style.cssText = `
            position: absolute;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            max-width: 520px;
            padding: 48px 40px;
            background: white;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.12);
        `;

        // User-facing Italian copy — left untouched (it's content, not code).
        box.innerHTML = `
            <div style="font-size: 2.5rem; margin-bottom: 16px;">👁️</div>
            <h2 style="margin: 0 0 12px 0; font-size: 1.4rem; color: #1e293b;">
                Calibrazione sguardo
            </h2>
            <p style="margin: 0 0 24px 0; color: #475569; font-size: 1rem; line-height: 1.6;">
                Apparirà una serie di <strong>9 punti rossi</strong> in posizioni diverse dello schermo.<br><br>
                Per ogni punto: <strong>fissa il pallino</strong> e tieni premuto <kbd style="background:#f1f5f9;border:1px solid #cbd5e1;border-radius:4px;padding:2px 8px;font-size:0.9rem;">SPAZIO</kbd> finché l'anello non si completa.
            </p>
            <p style="margin: 0; color: #94a3b8; font-size: 0.85rem;">
                Assicurati di essere seduto nella posizione che userai normalmente.<br>
                La calibrazione richiede circa 30 secondi.
            </p>
            <div style="margin-top: 32px; padding: 12px 24px; background: #f8fafc; border-radius: 8px; color: #64748b; font-size: 0.9rem;">
                Premi <kbd style="background:white;border:1px solid #cbd5e1;border-radius:4px;padding:2px 8px;">SPAZIO</kbd> per iniziare
            </div>
        `;

        this.overlay.appendChild(box);
    }

    /**
     * Render the next anchor dot, an SVG progress ring around it, a
     * "Point n / N" counter and a bottom-of-screen instruction label.
     * Calling this past the last anchor closes the overlay and invokes
     * the completion callback.
     *
     * @private
     */
    _showNextPoint() {
        this.overlay.innerHTML = '';
        // Termination condition: all anchors recorded → hand off to main.js
        if (this.currentIndex >= this.points.length) {
            this.overlay.style.display = 'none';
            this.onComplete();
            return;
        }

        const p = this.points[this.currentIndex];

        // Inline @keyframes for the pulsing dot — defined here so the
        // animation lives and dies with the overlay (no leftover global CSS)
        const style = document.createElement('style');
        style.textContent = '@keyframes _cpulse{0%,100%{transform:translate(-50%,-50%) scale(1)}50%{transform:translate(-50%,-50%) scale(1.2)}}';
        this.overlay.appendChild(style);

        // The red dot itself: pulses gently to attract the eye
        const dot = document.createElement('div');
        dot.style.cssText = `position:absolute;left:${p.x * 100}%;top:${p.y * 100}%;width:18px;height:18px;background:crimson;border-radius:50%;transform:translate(-50%,-50%);animation:_cpulse 0.9s ease-in-out infinite;`;
        this.overlay.appendChild(dot);

        // Build the SVG progress ring as a stroked circle. We start fully
        // empty (stroke-dashoffset = perimeter) and decrement the offset
        // towards 0 as samples accumulate, giving the impression of a
        // filling ring. The -90° rotation places the start of the stroke
        // at 12 o'clock instead of 3 o'clock.
        const R = 20, perim = Math.round(2 * Math.PI * R);
        this._ringPerim = perim;
        const ns = 'http://www.w3.org/2000/svg';
        const svg = document.createElementNS(ns, 'svg');
        svg.setAttribute('width', 52); svg.setAttribute('height', 52);
        svg.style.cssText = `position:absolute;left:${p.x * 100}%;top:${p.y * 100}%;transform:translate(-50%,-50%);pointer-events:none;`;
        const circle = document.createElementNS(ns, 'circle');
        circle.setAttribute('cx', 26); circle.setAttribute('cy', 26); circle.setAttribute('r', R);
        circle.setAttribute('fill', 'none'); circle.setAttribute('stroke', 'crimson');
        circle.setAttribute('stroke-width', 3);
        circle.setAttribute('stroke-dasharray', perim);
        circle.setAttribute('stroke-dashoffset', perim);
        circle.setAttribute('transform', 'rotate(-90 26 26)');
        svg.appendChild(circle);
        this.overlay.appendChild(svg);
        this._ringEl = circle;

        // Counter in the top-right corner: e.g. "Punto 3 / 9"
        const counter = document.createElement('div');
        counter.style.cssText = 'position:absolute;top:14px;right:20px;font-size:1rem;font-weight:600;color:#334155;';
        counter.innerText = `Punto ${this.currentIndex + 1} / ${this.points.length}`;
        this.overlay.appendChild(counter);

        // Bottom instruction label — mutated by updateProgress() as the user holds SPACE
        const label = document.createElement('p');
        label.innerText = 'Fissa il punto rosso e tieni premuto SPAZIO';
        label.style.cssText = 'position:absolute;bottom:20px;width:100%;text-align:center;font-weight:600;color:#334155;margin:0;';
        this.overlay.appendChild(label);
        this._labelEl = label;
    }

    /**
     * Animate the progress ring and the instruction label according to
     * how many calibration samples have been collected for the current
     * anchor. Driven externally by main.js, once per frame.
     *
     * @param {number} progress - In [0, 1]: 0 means no samples yet,
     *                            1 means the anchor is fully recorded.
     */
    updateProgress(progress) {
        if (!this._ringEl) return;
        // stroke-dashoffset shrinks linearly from perim → 0 as progress goes 0 → 1
        this._ringEl.setAttribute('stroke-dashoffset', (this._ringPerim * (1 - progress)).toFixed(1));
        if (this._labelEl) {
            this._labelEl.innerText = progress < 1
                ? `Tieni fermo… ${Math.round(progress * 100)}%`
                : 'Punto registrato!';
        }
    }

    /**
     * Return the current anchor's screen-pixel coordinates. main.js needs
     * them as the (screenX, screenY) target paired with each iris sample
     * fed to the TPS calibrator.
     *
     * @returns {{x:number,y:number}} Absolute pixel coordinates.
     */
    getNextPointCoords() {
        const p = this.points[this.currentIndex];
        return { x: p.x * window.innerWidth, y: p.y * window.innerHeight };
    }

    /**
     * Move on to the next anchor. Called by main.js after the multi-frame
     * sample buffer for the current anchor has been processed and stored
     * in the TPS calibrator.
     */
    advance() {
        this.currentIndex++;
        this._showNextPoint();
    }
}