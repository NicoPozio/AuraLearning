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

        this.currentAudio = new Audio();
        this.currentAudio.crossOrigin = "anonymous";

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

        let eyebrows = [];
        let headBone = null;

        this.model.traverse((child) => {
            if (child.isMesh) {
                child.castShadow = true;
                child.receiveShadow = true;

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
                if (n === 'mixamorighead' || n === 'head') headBone = child;
            }
        });

        if (headBone && eyebrows.length > 0) {
            eyebrows.forEach(eb => {
                eb.isSkinnedMesh = false;
                headBone.attach(eb);
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

                    // --- RISOLTO PROBLEMA ANIMAZIONE CHE SI FERMA ---
                    // Ora TUTTE le animazioni (inclusa SPEAKING) andranno in loop infinito
                    // Sarà la funzione speak() a fermarla esattamente quando l'audio si interrompe
                    action.setLoop(THREE.LoopRepeat, Infinity);
                    action.clampWhenFinished = false;

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

    update(dtSec) {
        if (this.mixer) this.mixer.update(dtSec);
        this.renderer.render(this.scene, this.camera);
    }

    async speak(text) {
        if (this.audioContext && this.audioContext.state === 'suspended') this.audioContext.resume();
        this.currentAudio.pause();
        
        try {
            console.log(`[AURA] Richiesta TTS al Proxy: "${text}"`);
            
            //Chiamata diretta al server locale
            const response = await fetch("http://localhost:8000/api/tts", {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                throw new Error(`Errore Proxy: HTTP ${response.status}`);
            }

            const blob = await response.blob();
            const audioUrl = URL.createObjectURL(blob);

            //Pulizia eventi precedenti
            this.currentAudio.onplay = null;
            this.currentAudio.onended = null;
            this.currentAudio.onerror = null;

            this.currentAudio.src = audioUrl;

            return new Promise((resolve) => {
                this.currentAudio.onplay = () => {
                    this.isSpeaking = true;
                    this.setState('SPEAKING');
                };

                // ECCO LA CORREZIONE: onended invece di onend!
                this.currentAudio.onended = () => {
                    this.isSpeaking = false;
                    this.setState('IDLE');
                    URL.revokeObjectURL(audioUrl);
                    resolve();
                };

                this.currentAudio.onerror = (e) => {
                    console.error("❌ [AURA] Errore riproduzione audio.");
                    this.isSpeaking = false;
                    this.setState('IDLE');
                    resolve();
                };

                this.currentAudio.play().catch((err) => {
                    console.error("❌ Autoplay bloccato dal browser!", err);
                    this.isSpeaking = false;
                    this.setState('IDLE');
                    resolve();
                });
            });

        } catch (error) {
            console.error("❌ [AURA] Fallimento API ElevenLabs:", error);
            this.isSpeaking = false;
            this.setState('IDLE');
        }
    }

    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
}