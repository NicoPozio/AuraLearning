/**
 * GazeEstimator: Estrazione delle feature oculari.
 * Implementa l'estrazione del centroide binoculare per massimizzare il rapporto segnale/rumore.
 */
export class GazeEstimator {
    static LEFT_IRIS = [474, 475, 476, 477];
    static RIGHT_IRIS = [469, 470, 471, 472];

    /**
     * Calcola la posizione media assoluta di entrambe le iridi nel frame della telecamera.
     * @param {Array} landmarks Landmark facciali forniti da MediaPipe.
     * @returns {Object} Vettore {x, y} assoluto normalizzato (0.0 - 1.0).
     */
    static getRobustGazeVector(landmarks) {
        if (!landmarks || landmarks.length < 478) return { x: 0, y: 0 };

        let sumX = 0, sumY = 0;
        
        for (let i = 0; i < 4; i++) {
            sumX += landmarks[this.LEFT_IRIS[i]].x + landmarks[this.RIGHT_IRIS[i]].x;
            sumY += landmarks[this.LEFT_IRIS[i]].y + landmarks[this.RIGHT_IRIS[i]].y;
        }

        // Il calcolo della media su 8 punti (4 per occhio) agisce come un 
        // filtro passa-basso spaziale, stabilizzando notevolmente la coordinata.
        return {
            x: sumX / 8,
            y: sumY / 8
        };
    }
}