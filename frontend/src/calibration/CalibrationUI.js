export class CalibrationUI {
    constructor(onComplete) {
        this.onComplete = onComplete;
        this.points = [
            { x: 0.1, y: 0.1 }, { x: 0.9, y: 0.1 },
            { x: 0.5, y: 0.5 },
            { x: 0.1, y: 0.9 }, { x: 0.9, y: 0.9 },
            { x: 0.5, y: 0.1 }, { x: 0.5, y: 0.9 }
        ];
        this.currentIndex = 0;
        this.overlay = this._createOverlay();
    }

    _createOverlay() {
        const div = document.createElement('div');
        div.style = "position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(255,255,255,0.9); z-index:9999; display:none; cursor:none;";
        document.body.appendChild(div);
        return div;
    }

    start() {
        this.overlay.style.display = 'block';
        this.currentIndex = 0;
        this._showNextPoint();
    }

    _showNextPoint() {
        this.overlay.innerHTML = '';
        if (this.currentIndex >= this.points.length) {
            this.overlay.style.display = 'none';
            this.onComplete();
            return;
        }

        const p = this.points[this.currentIndex];
        const dot = document.createElement('div');
        dot.style = `position:absolute; left:${p.x * 100}%; top:${p.y * 100}%; width:20px; height:20px; background:red; border-radius:50%; transform:translate(-50%,-50%); transition: all 0.3s;`;
        this.overlay.appendChild(dot);

        // Istruzione per l'utente
        const label = document.createElement('p');
        label.innerText = "Fissa il punto rosso e premi SPAZIO";
        label.style = "position:absolute; bottom:20px; width:100%; text-align:center; font-weight:bold;";
        this.overlay.appendChild(label);
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