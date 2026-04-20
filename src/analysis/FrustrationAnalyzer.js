export class FrustrationAnalyzer {
    constructor() {
        this.isCalibrating = false;
        this.isCalibrated = false;
        this.samples = [];
        this.baseline = {
            corrugator: { mean: 0, std: 1 },
            ear: { mean: 0, std: 1 },
            lipPress: { mean: 0, std: 1 }
        };
        
        this.stressAccumulator = 0.0;
        this.activationThreshold = 100.0; 
    }

    startCalibration() {
        this.isCalibrating = true;
        this.isCalibrated = false;
        this.samples = [];
    }

    processCalibrationSample(metrics) {
        if (!this.isCalibrating) return false;
        
        this.samples.push(metrics);
        
        // Calibrazione completata dopo 120 frame (circa 2 secondi)
        if (this.samples.length >= 120) {
            const calculateStats = (key) => {
                const mean = this.samples.reduce((s, m) => s + m[key], 0) / 120;
                const variance = this.samples.reduce((s, m) => s + Math.pow(m[key] - mean, 2), 0) / 120;
                return { mean, std: Math.sqrt(variance) || 0.001 };
            };

            this.baseline.corrugator = calculateStats('corrugator');
            this.baseline.ear = calculateStats('ear');
            this.baseline.lipPress = calculateStats('lipPress');
            
            this.isCalibrating = false;
            this.isCalibrated = true;
            return true;
        }
        return false;
    }

    update(metrics, dtSec) {
        if (!this.isCalibrated) return null;

        // Calcolo dello Z-Score: (Valore - Media) / Deviazione Standard
        // Nota: AU4 (Corrugatore), EAR e LipPress diminuiscono in valore numerico durante la contrazione.
        // Invertiamo il segno affinché una contrazione risulti in uno Z-Score positivo.
        const zCorrugator = (this.baseline.corrugator.mean - metrics.corrugator) / this.baseline.corrugator.std;
        const zEar = (this.baseline.ear.mean - metrics.ear) / this.baseline.ear.std;
        const zLip = (this.baseline.lipPress.mean - metrics.lipPress) / this.baseline.lipPress.std;

        // Filtro passa-basso logico: consideriamo solo attivazioni superiori a 1.5 sigma (rumore di fondo)
        const actCorr = zCorrugator > 1.5 ? zCorrugator : 0;
        const actEar = zEar > 1.5 ? zEar : 0;
        const actLip = zLip > 1.5 ? zLip : 0;

        // Integrazione dell'equazione differenziale
        const stressDelta = (actCorr * 1.5) + actEar + actLip;

        if (stressDelta > 0) {
            this.stressAccumulator += (stressDelta * 15.0 * dtSec);
        } else {
            this.stressAccumulator -= (20.0 * dtSec); // Decadimento
        }

        this.stressAccumulator = Math.max(0, Math.min(this.stressAccumulator, this.activationThreshold * 1.2));

        return {
            zCorrugator, zEar, zLip,
            isFrustrated: this.stressAccumulator >= this.activationThreshold,
            stressPercentage: (this.stressAccumulator / this.activationThreshold) * 100
        };
    }
}