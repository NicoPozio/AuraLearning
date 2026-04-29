import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// Ossa gambe — congelate al caricamento
const LEG_BONE_NAMES = new Set([
    'Bone010_43', 'Bone014_42', 'Bone015_41', 'Bone017_40',
    'Bone011_47', 'Bone012_46', 'Bone013_45', 'Bone016_44',
]);

export class ECAController {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.statusEl = document.getElementById('eca-status');

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xa0d8ef);

        this.camera = new THREE.PerspectiveCamera(
            45, this.container.clientWidth / this.container.clientHeight, 0.1, 100
        );
        this.camera.position.set(0, 1.2, 3.5);

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.container.appendChild(this.renderer.domElement);

        this.scene.add(new THREE.GridHelper(10, 10, 0xffffff, 0x888888));
        this.scene.add(new THREE.AmbientLight(0xffffff, 2.5));
        const dirLight = new THREE.DirectionalLight(0xffffff, 2);
        dirLight.position.set(2, 5, 2);
        this.scene.add(dirLight);

        this.model = null;
        this.jawBone = null;
        this.isSpeaking = false;
        this._idleT = 0;
        this._spine = null;
        this._armL = null;
        this._armR = null;
        this._frozenLegs = [];

        this._armLBaseZ = 0;
        this._armRBaseZ = 0;

        window.addEventListener('resize', () => this.onWindowResize());
    }

    async loadModel(modelUrl) {
        this.statusEl.innerText = "Caricamento...";
        const loader = new GLTFLoader();

        return new Promise((resolve, reject) => {
            loader.load(modelUrl, (gltf) => {
                this.model = gltf.scene;
                this.model.rotation.set(0, 0, 0);
                this.model.position.set(0, 0, 0);
                this.model.scale.set(1, 1, 1);
                this.scene.add(this.model);

                // Scala
                this.model.updateMatrixWorld(true);
                const box = new THREE.Box3().setFromObject(this.model);
                const size = new THREE.Vector3();
                box.getSize(size);
                const scale = 1.8 / (size.y || 1);
                this.model.scale.set(scale, scale, scale);

                // Posizionamento
                this.model.updateMatrixWorld(true);
                const box2 = new THREE.Box3().setFromObject(this.model);
                const center = new THREE.Vector3();
                box2.getCenter(center);
                this.model.position.y = -box2.min.y;
                this.model.position.x = -center.x;
                this.model.position.z = -center.z;

                // Mappa ossa sicure
                this.model.traverse((child) => {
                    if (!child.isBone) return;
                    const n = child.name;
                    if (n === 'Bone_39') this._spine = child;
                    if (n === 'Bone002_20') this._armL = child;
                    if (n === 'Bone006_36') this._armR = child;
                    if (n === 'Bone047_37') this.jawBone = child;
                });

                requestAnimationFrame(() => {
                    this._frozenLegs = [];
                    this.model.traverse((child) => {
                        if (!child.isBone || !LEG_BONE_NAMES.has(child.name)) return;
                        this._frozenLegs.push({ bone: child, q: child.quaternion.clone() });
                    });

                    if (this._armL) this._armLBaseZ = this._armL.rotation.z;
                    if (this._armR) this._armRBaseZ = this._armR.rotation.z;

                    // Stampa rotazione base della mascella per capire l'asse corretto
                    if (this.jawBone) {
                        const r = this.jawBone.rotation;
                        console.log(`Jaw base rotation → x:${r.x.toFixed(3)} y:${r.y.toFixed(3)} z:${r.z.toFixed(3)}`);
                    }

                    console.log(`Gambe congelate: ${this._frozenLegs.length}`);
                    this.statusEl.innerText = "ECA Pronto";
                    resolve();
                });

            }, undefined, reject);
        });
    }

    _lockLegs() {
        for (const { bone, q } of this._frozenLegs) {
            bone.quaternion.copy(q);
        }
    }

    _updateIdle(dt) {
        this._idleT += dt;
        const t = this._idleT;
        const breath = Math.sin(t * 1.5) * 0.012;
        const sway = Math.cos(t * 0.8) * 0.008;

        if (this._spine) {
            this._spine.rotation.x = breath;
            this._spine.rotation.z = sway;
        }
        if (this._armL) this._armL.rotation.z = this._armLBaseZ + sway * 2;
        if (this._armR) this._armR.rotation.z = this._armRBaseZ - sway * 2;
    }

    _updateJaw() {
        if (!this.jawBone) return;

        if (this.isSpeaking) {
            // Due frequenze sovrapposte per un parlato naturale
            const vol = Math.abs(Math.sin(this._idleT * 18)) * 0.6
                + Math.abs(Math.sin(this._idleT * 7)) * 0.4;

            // Ampiezza aumentata: 0.25 → 0.55 (ben visibile anche con la barba)
            const target = vol * 0.55;

            this.jawBone.rotation.x = THREE.MathUtils.lerp(
                this.jawBone.rotation.x, target, 0.4
            );
        } else {
            this.jawBone.rotation.x = THREE.MathUtils.lerp(
                this.jawBone.rotation.x, 0, 0.15
            );
        }
    }

    update(dtSec) {
        this._updateIdle(dtSec);
        this._lockLegs();
        this._updateJaw();
        this.renderer.render(this.scene, this.camera);
    }

    speak(text) {
        window.speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(text);
        u.lang = 'it-IT';
        u.rate = 0.95;
        u.pitch = 1.0;
        u.onstart = () => { this.isSpeaking = true; this.statusEl.innerText = "Parlando..."; };
        u.onend = () => { this.isSpeaking = false; this.statusEl.innerText = "ECA Pronto"; };
        u.onerror = () => { this.isSpeaking = false; this.statusEl.innerText = "ECA Pronto"; };
        window.speechSynthesis.speak(u);
    }

    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
}
