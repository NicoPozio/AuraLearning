import * as THREE from 'three';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';

export class ECAController {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.statusEl = document.getElementById('eca-status');

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xa0d8ef);

        this.camera = new THREE.PerspectiveCamera(
            45, this.container.clientWidth / this.container.clientHeight, 0.1, 500
        );
        this.camera.position.set(0, 1.2, 3.5);

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.shadowMap.enabled = true;
        this.container.appendChild(this.renderer.domElement);

        this.scene.add(new THREE.GridHelper(10, 10, 0xffffff, 0x888888));
        this.scene.add(new THREE.AmbientLight(0xffffff, 2.5));
        const dirLight = new THREE.DirectionalLight(0xffffff, 2);
        dirLight.position.set(2, 5, 2);
        this.scene.add(dirLight);

        this.model = null;
        this.mixer = null;
        this.clock = new THREE.Clock();

        this.actions = {
            IDLE: null,
            LISTENING: null,
            THINKING: null,
            SPEAKING: null,
        };

        this.currentAction = null;
        this.currentState = 'NONE';
        this.isSpeaking = false;

        this.jawBone = null;
        this.headBone = null;

        this.faceMesh = null;
        this.mouthOpenIndex = -1;

        this._idleT = 0;

        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 256;
        this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);

        window.addEventListener('resize', () => this.onWindowResize());
    }

    async loadModel(modelUrl) {
        this.statusEl.innerText = "Caricamento...";
        const fbxLoader = new FBXLoader();
        const textureLoader = new THREE.TextureLoader();

        const path = './models/textures/';

        const bodyColor = textureLoader.load(path + 'Professor_John_Body_part_BaseColor.png');
        bodyColor.colorSpace = THREE.SRGBColorSpace;
        const bodyNormal = textureLoader.load(path + 'Professor_John_Body_part_Normal.png');

        const clothColor = textureLoader.load(path + 'Professor_John_Cloth_BaseColor.png');
        clothColor.colorSpace = THREE.SRGBColorSpace;
        const clothNormal = textureLoader.load(path + 'Professor_John_Cloth_Normal.png');

        const hairColor = textureLoader.load(path + 'Professor_John_Hair_BaseColor.png');
        hairColor.colorSpace = THREE.SRGBColorSpace;
        const hairNormal = textureLoader.load(path + 'Professor_John_Hair_Normal.png');

        const shoeColor = textureLoader.load(path + 'Professor_John_Shoes_BaseColor.png');
        shoeColor.colorSpace = THREE.SRGBColorSpace;
        const shoeNormal = textureLoader.load(path + 'Professor_John_Shoes_Normal.png');

        const fbx = await new Promise((res, rej) =>
            fbxLoader.load('./models/personaggio.fbx', res, undefined, rej)
        );

        this.model = fbx;
        this.scene.add(this.model);

        this.model.updateMatrixWorld(true);
        const box = new THREE.Box3().setFromObject(this.model);
        const size = new THREE.Vector3();
        box.getSize(size);
        const scale = 1.8 / (size.y || 1);
        this.model.scale.set(scale, scale, scale);

        this.model.updateMatrixWorld(true);
        const box2 = new THREE.Box3().setFromObject(this.model);
        const center = new THREE.Vector3();
        box2.getCenter(center);
        this.model.position.y = -box2.min.y;
        this.model.position.x = -center.x;
        this.model.position.z = -center.z;

        const modelHeight = 1.8;
        const targetY = modelHeight * 0.5;
        this.camera.position.set(0, targetY, 2.8);
        this.camera.lookAt(0, targetY, 0);

        this.scene.children
            .filter(c => c instanceof THREE.GridHelper)
            .forEach(g => this.scene.remove(g));
        this.scene.add(new THREE.GridHelper(10, 10, 0xffffff, 0x888888));

        this.mixer = new THREE.AnimationMixer(this.model);

        // --- NUOVO: EVENTO FINE ANIMAZIONE ---
        // Questo listener scatta quando un'animazione non in loop finisce.
        this.mixer.addEventListener('finished', (e) => {
            // Se l'animazione che ha finito è quella del parlato, e l'audio sta ancora andando...
            if (e.action === this.actions['SPEAKING'] && this.isSpeaking) {
                // Sfumiamo dolcemente il corpo verso la posa IDLE
                this.actions['SPEAKING'].fadeOut(0.4);
                this.actions['IDLE'].reset().fadeIn(0.4).play();
                this.currentAction = this.actions['IDLE'];
            }
        });

        let eyebrows = [];

        this.model.traverse((child) => {
            if (child.isMesh) {
                child.castShadow = true;
                child.receiveShadow = true;

                if (child.morphTargetDictionary) {
                    this.faceMesh = child;
                    const keys = Object.keys(child.morphTargetDictionary);
                    for (let key of keys) {
                        if (key.toLowerCase().includes('mouth') || key.toLowerCase().includes('jaw') || key.toLowerCase() === 'a') {
                            this.mouthOpenIndex = child.morphTargetDictionary[key];
                            break;
                        }
                    }
                }

                if (child.name.toLowerCase().includes('eyebrow')) {
                    eyebrows.push(child);
                }

                if (child.material) {
                    const materials = Array.isArray(child.material) ? child.material : [child.material];

                    materials.forEach(mat => {
                        if (mat.shininess !== undefined) mat.shininess = 5;
                        if (mat.specular) mat.specular = new THREE.Color(0x111111);

                        const name = child.name.toLowerCase();

                        if (name.includes('body') || name.includes('head') || name.includes('face') || name.includes('skin') || name.includes('hand') || name.includes('arm')) {
                            mat.map = bodyColor;
                            mat.normalMap = bodyNormal;
                        }
                        else if (name.includes('shirt') || name.includes('pants') || name.includes('clothes') || name.includes('leg') || name.includes('bottom') || name.includes('top')) {
                            mat.map = clothColor;
                            mat.normalMap = clothNormal;
                        }
                        else if (name.includes('shoe') || name.includes('foot') || name.includes('sneaker')) {
                            mat.map = shoeColor;
                            mat.normalMap = shoeNormal;
                        }
                        else if (name.includes('hair') || name.includes('eyebrow') || name.includes('beard') || name.includes('moustache')) {
                            mat.color.setHex(0x4a3b32);
                            mat.map = null;
                        }

                        mat.needsUpdate = true;
                    });
                }
            }

            if (child.isBone) {
                const n = child.name.toLowerCase();

                if (n.includes('jaw')) {
                    this.jawBone = child;
                }
                if (n === 'mixamorighead' || n === 'head') {
                    this.headBone = child;
                }
            }
        });

        if (this.headBone && eyebrows.length > 0) {
            eyebrows.forEach(eb => {
                eb.isSkinnedMesh = false;
                this.headBone.attach(eb);
            });
        }

        const animFiles = {
            IDLE: './models/idle.fbx',
            LISTENING: './models/nodding.fbx',
            THINKING: './models/thinking.fbx',
            SPEAKING: './models/talking.fbx',
        };

        for (const [state, path] of Object.entries(animFiles)) {
            try {
                const animFbx = await new Promise((res, rej) =>
                    fbxLoader.load(path, res, undefined, rej)
                );
                if (animFbx.animations && animFbx.animations.length > 0) {
                    const clip = animFbx.animations[0];
                    const action = this.mixer.clipAction(clip);

                    // --- NUOVA LOGICA DEI LOOP ---
                    if (state === 'SPEAKING') {
                        // Se è l'animazione di parlato, la esegue 1 SOLA VOLTA e si ferma
                        action.setLoop(THREE.LoopOnce, 1);
                        action.clampWhenFinished = true;
                    } else {
                        // Tutte le altre (Idle, Thinking) vanno in loop all'infinito
                        action.setLoop(THREE.LoopRepeat, Infinity);
                        action.clampWhenFinished = false;
                    }

                    this.actions[state] = action;
                }
            } catch (e) {
                console.warn(`[ECA] File non trovato: ${path}`);
            }
        }

        this.setState('IDLE');
    }

    setState(newState) {
        if (this.currentState === newState) return;
        this.currentState = newState;

        if (newState === 'IDLE') this.statusEl.innerText = "ECA Pronto";
        else if (newState === 'SPEAKING') this.statusEl.innerText = "ECA Parlando...";
        else if (newState === 'LISTENING') this.statusEl.innerText = "ECA In Ascolto...";
        else if (newState === 'THINKING') this.statusEl.innerText = "ECA Pensando...";

        const spinner = document.getElementById('thinking-spinner');
        if (spinner) spinner.style.display = (newState === 'THINKING') ? 'block' : 'none';

        const nextAction = this.actions[newState] ?? this.actions['IDLE'];
        if (nextAction && nextAction !== this.currentAction) {
            if (this.currentAction) this.currentAction.fadeOut(0.3);
            nextAction.reset().fadeIn(0.3).play();
            this.currentAction = nextAction;
        }
    }

    _updateJawVolumetric() {
        if (this.isSpeaking) {
            this.analyser.getByteFrequencyData(this.dataArray);
            let sum = 0;
            for (let i = 0; i < this.dataArray.length; i++) sum += this.dataArray[i];
            const volume = sum / this.dataArray.length;

            if (this.jawBone) {
                const targetRotation = (volume / 255) * 0.7;
                this.jawBone.rotation.x = THREE.MathUtils.lerp(this.jawBone.rotation.x, targetRotation, 0.4);
            } else if (this.faceMesh && this.mouthOpenIndex !== -1) {
                const targetMorph = (volume / 255) * 1.5;
                const currentMorph = this.faceMesh.morphTargetInfluences[this.mouthOpenIndex];
                this.faceMesh.morphTargetInfluences[this.mouthOpenIndex] = THREE.MathUtils.lerp(currentMorph, targetMorph, 0.4);
            } else if (this.headBone) {
                const nodAmount = (volume / 255) * 0.4;
                this.headBone.rotateX(nodAmount);
            }
        } else {
            if (this.jawBone) {
                this.jawBone.rotation.x = THREE.MathUtils.lerp(this.jawBone.rotation.x, 0, 0.15);
            } else if (this.faceMesh && this.mouthOpenIndex !== -1) {
                const currentMorph = this.faceMesh.morphTargetInfluences[this.mouthOpenIndex];
                this.faceMesh.morphTargetInfluences[this.mouthOpenIndex] = THREE.MathUtils.lerp(currentMorph, 0, 0.2);
            }
        }
    }

    update(dtSec) {
        if (this.mixer) this.mixer.update(dtSec);
        this._updateJawVolumetric();
        this.renderer.render(this.scene, this.camera);
    }

    speak(text) {
        if (this.audioContext.state === 'suspended') this.audioContext.resume();

        window.speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(text);
        u.lang = 'it-IT';
        u.rate = 0.95;
        u.pitch = 1.0;

        u.onstart = () => { this.isSpeaking = true; this.setState('SPEAKING'); };
        u.onend = () => { this.isSpeaking = false; this.setState('IDLE'); };
        u.onerror = () => { this.isSpeaking = false; this.setState('IDLE'); };

        window.speechSynthesis.speak(u);
    }

    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
}