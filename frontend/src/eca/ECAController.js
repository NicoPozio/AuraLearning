import * as THREE from 'three';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';

/**
 * Embodied Conversational Agent (ECA) controller.
 *
 * Wraps a Three.js scene that displays a rigged 3D avatar with four
 * animation states (IDLE, LISTENING, THINKING, SPEAKING) and exposes a
 * high-level speak() method that synchronises the animation state
 * machine with TTS audio playback.
 *
 * Lifecycle:
 *   constructor()  — set up the Three.js scene, camera, lights, renderer
 *   loadModel()    — load the FBX rig, textures and four animation clips
 *   update(dt)     — per-frame tick, advances the animation mixer
 *   setState(s)    — transition between animation states with crossfade
 *   speak(text)    — send text to the TTS proxy and play the audio
 */
export class ECAController {

    /**
     * Build the Three.js scene used to render the avatar. The model is
     * loaded asynchronously by loadModel(); the constructor only sets
     * up everything that does not depend on the rig.
     *
     * @param {string} containerId - DOM id of the <div> hosting the canvas.
     */
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.statusEl = document.getElementById('eca-status');

        this.scene = new THREE.Scene();
        // Sky-blue background: neutral with respect to skin tones.
        this.scene.background = new THREE.Color(0xa0d8ef);

        // 45° FOV gives a portrait-style framing. Camera position is
        // overwritten in loadModel() once the model size is known.
        this.camera = new THREE.PerspectiveCamera(
            45, this.container.clientWidth / this.container.clientHeight, 0.1, 500
        );
        this.camera.position.set(0, 1.2, 3.5);

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.shadowMap.enabled = true;
        this.container.appendChild(this.renderer.domElement);

        // GridHelper acts as a visual floor reference while the avatar
        // loads. Strong ambient light avoids flat shadows on the skin;
        // a directional key light adds shape.
        this.scene.add(new THREE.GridHelper(10, 10, 0xffffff, 0x888888));
        this.scene.add(new THREE.AmbientLight(0xffffff, 2.5));
        const dirLight = new THREE.DirectionalLight(0xffffff, 2);
        dirLight.position.set(2, 5, 2);
        this.scene.add(dirLight);

        // Animation state.
        this.model = null;         // root Object3D of the loaded FBX
        this.mixer = null;         // AnimationMixer driving the clips
        this.actions = {           // pre-allocated slots for the 4 actions
            IDLE: null,
            LISTENING: null,
            THINKING: null,
            SPEAKING: null,
        };
        this.currentAction = null; // currently playing AnimationAction
        this.currentState = 'NONE';

        // Single audio element reused for every utterance. crossOrigin
        // is set so we can stream audio served by the FastAPI backend
        // without CORS issues.
        this.currentAudio = new Audio();
        this.currentAudio.crossOrigin = "anonymous";

        // Keep the canvas aspect ratio consistent on viewport resize.
        window.addEventListener('resize', () => this.onWindowResize());
    }

    /**
     * Load the FBX rig, its PBR textures and the four animation clips,
     * then fit the camera around the resized model and enter IDLE.
     *
     * Texture-to-mesh binding is heuristic: it inspects each mesh name
     * and matches it against keyword sets (body / clothes / shoes).
     * Hair meshes are flat-coloured because the source rig ships with
     * sparse hair textures that look noisy on a stylised character.
     */
    async loadModel() {
        this.statusEl.innerText = "Caricamento...";
        const fbxLoader = new FBXLoader();
        const textureLoader = new THREE.TextureLoader();

        const path = './models/textures/';

        // Texture pre-loading. BaseColor maps are tagged sRGB so
        // Three.js applies the correct gamma conversion; Normal maps
        // stay in linear space (default).
        const bodyColor = textureLoader.load(path + 'Professor_John_Body_part_BaseColor.png');
        bodyColor.colorSpace = THREE.SRGBColorSpace;
        const bodyNormal = textureLoader.load(path + 'Professor_John_Body_part_Normal.png');

        const clothColor = textureLoader.load(path + 'Professor_John_Cloth_BaseColor.png');
        clothColor.colorSpace = THREE.SRGBColorSpace;
        const clothNormal = textureLoader.load(path + 'Professor_John_Cloth_Normal.png');

        const shoeColor = textureLoader.load(path + 'Professor_John_Shoes_BaseColor.png');
        shoeColor.colorSpace = THREE.SRGBColorSpace;
        const shoeNormal = textureLoader.load(path + 'Professor_John_Shoes_Normal.png');

        // FBX rig load (wrapped in a Promise for await-friendliness).
        const fbx = await new Promise((res, rej) =>
            fbxLoader.load('./models/personaggio.fbx', res, undefined, rej)
        );

        this.model = fbx;
        this.scene.add(this.model);

        // Auto-fit: rescale the model to a target height of 1.8 m.
        this.model.updateMatrixWorld(true);
        const box = new THREE.Box3().setFromObject(this.model);
        const size = new THREE.Vector3();
        box.getSize(size);
        const scale = 1.8 / (size.y || 1);
        this.model.scale.set(scale, scale, scale);

        // Auto-centre: re-measure after scaling and shift the model so
        // its feet touch y=0 and its centre sits on the X/Z axes.
        this.model.updateMatrixWorld(true);
        const box2 = new THREE.Box3().setFromObject(this.model);
        const center = new THREE.Vector3();
        box2.getCenter(center);
        this.model.position.y = -box2.min.y;
        this.model.position.x = -center.x;
        this.model.position.z = -center.z;

        // Re-aim the camera at the torso, with a slight head framing.
        const modelHeight = 1.8;
        const targetY = modelHeight * 0.5;
        this.camera.position.set(0, targetY, 2.8);
        this.camera.lookAt(0, targetY, 0);

        // Reset the floor grid: remove the helper added in the
        // constructor and add a fresh one positioned under the resized
        // model.
        this.scene.children
            .filter(c => c instanceof THREE.GridHelper)
            .forEach(g => this.scene.remove(g));
        this.scene.add(new THREE.GridHelper(10, 10, 0xffffff, 0x888888));

        this.mixer = new THREE.AnimationMixer(this.model);

        // Collected during the traversal below for the eyebrow
        // reparenting step.
        let eyebrows = [];
        let headBone = null;

        // Walk the scene graph: assign textures, collect eyebrows and
        // the head bone.
        this.model.traverse((child) => {
            if (child.isMesh) {
                child.castShadow = true;
                child.receiveShadow = true;

                // Eyebrows are reparented later so they follow the head
                // rotation instead of staying anchored to the root bone.
                if (child.name.toLowerCase().includes('eyebrow')) {
                    eyebrows.push(child);
                }

                if (child.material) {
                    // FBX meshes may carry one or many materials — normalise to an array.
                    const materials = Array.isArray(child.material) ? child.material : [child.material];

                    materials.forEach(mat => {
                        // Tone down the default Phong shininess: skin
                        // and cloth should not behave like polished
                        // plastic.
                        if (mat.shininess !== undefined) mat.shininess = 5;
                        if (mat.specular) mat.specular = new THREE.Color(0x111111);

                        const name = child.name.toLowerCase();

                        // Heuristic texture binding by mesh name.
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
                            // Hair meshes use a flat brown colour.
                            mat.color.setHex(0x4a3b32);
                            mat.map = null;
                        }

                        // Required after replacing maps or colours on
                        // an existing material.
                        mat.needsUpdate = true;
                    });
                }
            }

            // Capture the head bone for the eyebrow reparenting step.
            if (child.isBone) {
                const n = child.name.toLowerCase();
                if (n === 'mixamorighead' || n === 'head') headBone = child;
            }
        });

        // Eyebrow reparenting. The rigging in the FBX leaves the
        // eyebrows under the root bone, which makes them stay still
        // while the head rotates. Re-attaching them to the head bone
        // keeps them visually glued to the face.
        if (headBone && eyebrows.length > 0) {
            eyebrows.forEach(eb => {
                eb.isSkinnedMesh = false;
                headBone.attach(eb);
            });
        }

        // Load the four animation clips.
        const animFiles = {
            IDLE:      './models/idle.fbx',
            LISTENING: './models/nodding.fbx',
            THINKING:  './models/thinking.fbx',
            SPEAKING:  './models/talking.fbx',
        };

        for (const [state, filePath] of Object.entries(animFiles)) {
            try {
                const animFbx = await new Promise((res, rej) =>
                    fbxLoader.load(filePath, res, undefined, rej)
                );
                if (animFbx.animations && animFbx.animations.length > 0) {
                    const clip = animFbx.animations[0];
                    const action = this.mixer.clipAction(clip);

                    // All clips loop indefinitely (including SPEAKING).
                    // speak() is responsible for stopping the SPEAKING
                    // loop exactly when the audio playback ends.
                    action.setLoop(THREE.LoopRepeat, Infinity);
                    action.clampWhenFinished = false;

                    this.actions[state] = action;
                }
            } catch (e) {
                // A missing clip is non-fatal: the avatar simply won't
                // react for that specific state.
                console.warn(`[ECA] File non trovato: ${filePath}`);
            }
        }

        this.setState('IDLE');
    }

    /**
     * Transition the avatar to a new animation state with a 0.3 s
     * crossfade, and update the status badge plus the "thinking"
     * spinner overlay accordingly.
     *
     * No-op when the new state equals the current one, which avoids
     * re-triggering the same crossfade on every frame.
     *
     * @param {'IDLE'|'LISTENING'|'THINKING'|'SPEAKING'} newState
     */
    setState(newState) {
        if (this.currentState === newState) return;
        this.currentState = newState;

        if (newState === 'IDLE')           this.statusEl.innerText = "ECA Pronto";
        else if (newState === 'SPEAKING')  this.statusEl.innerText = "ECA Parlando...";
        else if (newState === 'LISTENING') this.statusEl.innerText = "ECA In Ascolto...";
        else if (newState === 'THINKING')  this.statusEl.innerText = "ECA Pensando...";

        const spinner = document.getElementById('thinking-spinner');
        if (spinner) spinner.style.display = (newState === 'THINKING') ? 'block' : 'none';

        // Fall back to IDLE if the requested action wasn't loaded.
        const nextAction = this.actions[newState] ?? this.actions['IDLE'];
        if (nextAction && nextAction !== this.currentAction) {
            if (this.currentAction) this.currentAction.fadeOut(0.3);
            nextAction.reset().fadeIn(0.3).play();
            this.currentAction = nextAction;
        }
    }

    /**
     * Per-frame tick. Advances the animation mixer and re-renders the
     * scene. Called from the main render loop.
     *
     * @param {number} dtSec - Elapsed time since the previous call.
     */
    update(dtSec) {
        if (this.mixer) this.mixer.update(dtSec);
        this.renderer.render(this.scene, this.camera);
    }

    /**
     * Synthesize and play `text` via the FastAPI TTS proxy. On HTTP 401
     * (ElevenLabs quota exhausted) or any network failure, falls back
     * to the browser's native SpeechSynthesis. Resolves when playback
     * ends (or on error).
     *
     * @param {string} text - Phrase to speak.
     * @returns {Promise<void>}
     */
    async speak(text) {
        // Cancel any audio in progress (both ElevenLabs and browser).
        this.currentAudio.pause();
        window.speechSynthesis.cancel();

        try {
            console.log(`[AURA] Richiesta TTS al proxy: "${text}"`);

            const response = await fetch("http://localhost:8000/api/tts", {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                if (response.status === 401) {
                    console.warn("[AURA] Crediti ElevenLabs esauriti (HTTP 401). Attivo fallback browser.");
                    return this._speakWithBrowserFallback(text);
                }
                throw new Error(`Errore proxy: HTTP ${response.status}`);
            }

            const blob = await response.blob();
            const audioUrl = URL.createObjectURL(blob);

            // Clean previous handlers.
            this.currentAudio.onplay = null;
            this.currentAudio.onended = null;
            this.currentAudio.onerror = null;

            this.currentAudio.src = audioUrl;

            return new Promise((resolve) => {
                this.currentAudio.onplay = () => {
                    this.setState('SPEAKING');
                };
                this.currentAudio.onended = () => {
                    this.setState('IDLE');
                    URL.revokeObjectURL(audioUrl);
                    resolve();
                };
                this.currentAudio.onerror = () => {
                    console.error("[AURA] Errore riproduzione audio ElevenLabs.");
                    this.setState('IDLE');
                    resolve();
                };
                this.currentAudio.play().catch((err) => {
                    console.error("[AURA] Autoplay bloccato dal browser:", err);
                    this.setState('IDLE');
                    resolve();
                });
            });

        } catch (error) {
            // Network or server failure: fall back to native TTS.
            console.error("[AURA] Fallimento API ElevenLabs:", error);
            console.warn("[AURA] Attivo fallback vocale del browser.");
            return this._speakWithBrowserFallback(text);
        }
    }

    /**
     * Native browser SpeechSynthesis fallback. Keeps the avatar
     * animation in sync with the synthesised audio.
     */
    _speakWithBrowserFallback(text) {
        return new Promise((resolve) => {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'it-IT';

            // Pick a default Italian voice if available.
            const voices = window.speechSynthesis.getVoices();
            const italianVoice = voices.find(v => v.lang.startsWith('it'));
            if (italianVoice) utterance.voice = italianVoice;

            utterance.onstart = () => { this.setState('SPEAKING'); };
            utterance.onend   = () => { this.setState('IDLE'); resolve(); };
            utterance.onerror = (e) => {
                console.error("[AURA] Fallback browser fallito:", e);
                this.setState('IDLE');
                resolve();
            };

            window.speechSynthesis.speak(utterance);
        });
    }

    /**
     * Recompute the camera aspect ratio and renderer size after a
     * viewport change, so the avatar canvas always fits its container
     * without distortion. Bound to window's "resize" event in the
     * constructor.
     */
    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
}