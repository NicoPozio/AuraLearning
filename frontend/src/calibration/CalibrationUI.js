export class CalibrationUI {
    constructor(onComplete) {
        this.onComplete = onComplete;
        this.points = [
            { x: 0.1, y: 0.1 }, { x: 0.9, y: 0.1 },
            { x: 0.1, y: 0.9 }, { x: 0.9, y: 0.9 },
            { x: 0.5, y: 0.5 },
            { x: 0.5, y: 0.1 }, { x: 0.5, y: 0.9 },
            { x: 0.25, y: 0.65 }, { x: 0.75, y: 0.65 },
        ];
        this.currentIndex = 0;
        this._phase   = 'intro'; // 'intro' | 'dots'
        this.overlay  = this._createOverlay();
        this._ringEl  = null;
        this._ringPerim = 0;
        this._labelEl = null;
    }

    _createOverlay() {
        const div = document.createElement('div');
        div.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;background:rgba(255,255,255,0.95);z-index:9999;display:none;cursor:none;font-family:system-ui,sans-serif;';
        document.body.appendChild(div);
        return div;
    }

    start() {
        this.overlay.style.display = 'block';
        this._phase = 'intro';
        this.currentIndex = 0;
        this._showIntro();
    }

    // Chiamato da main.js su keydown Space
    onSpaceDown() {
        if (this._phase === 'intro') {
            this._phase = 'dots';
            this._showNextPoint();
        }
        // Se siamo in 'dots', la raccolta multi-frame è gestita da main.js
    }

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

    _showNextPoint() {
        this.overlay.innerHTML = '';
        if (this.currentIndex >= this.points.length) {
            this.overlay.style.display = 'none';
            this.onComplete();
            return;
        }

        const p = this.points[this.currentIndex];

        const style = document.createElement('style');
        style.textContent = '@keyframes _cpulse{0%,100%{transform:translate(-50%,-50%) scale(1)}50%{transform:translate(-50%,-50%) scale(1.2)}}';
        this.overlay.appendChild(style);

        const dot = document.createElement('div');
        dot.style.cssText = `position:absolute;left:${p.x*100}%;top:${p.y*100}%;width:18px;height:18px;background:crimson;border-radius:50%;transform:translate(-50%,-50%);animation:_cpulse 0.9s ease-in-out infinite;`;
        this.overlay.appendChild(dot);

        const R = 20, perim = Math.round(2 * Math.PI * R);
        this._ringPerim = perim;
        const ns = 'http://www.w3.org/2000/svg';
        const svg = document.createElementNS(ns, 'svg');
        svg.setAttribute('width', 52); svg.setAttribute('height', 52);
        svg.style.cssText = `position:absolute;left:${p.x*100}%;top:${p.y*100}%;transform:translate(-50%,-50%);pointer-events:none;`;
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

        const counter = document.createElement('div');
        counter.style.cssText = 'position:absolute;top:14px;right:20px;font-size:1rem;font-weight:600;color:#334155;';
        counter.innerText = `Punto ${this.currentIndex + 1} / ${this.points.length}`;
        this.overlay.appendChild(counter);

        const label = document.createElement('p');
        label.innerText = 'Fissa il punto rosso e tieni premuto SPAZIO';
        label.style.cssText = 'position:absolute;bottom:20px;width:100%;text-align:center;font-weight:600;color:#334155;margin:0;';
        this.overlay.appendChild(label);
        this._labelEl = label;
    }

    updateProgress(progress) {
        if (!this._ringEl) return;
        this._ringEl.setAttribute('stroke-dashoffset', (this._ringPerim * (1 - progress)).toFixed(1));
        if (this._labelEl) {
            this._labelEl.innerText = progress < 1
                ? `Tieni fermo… ${Math.round(progress * 100)}%`
                : 'Punto registrato!';
        }
    }

    getNextPointCoords() {
        const p = this.points[this.currentIndex];
        return { x: p.x * window.innerWidth, y: p.y * window.innerHeight };
    }

    advance() {
        this.currentIndex++;
        this._showNextPoint();
    }
}