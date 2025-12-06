// front-end JS — front-only (aucune connexion au backend dans ce fichier)

/*
 * Le code suivant est strictement pour le navigateur. Le serveur de
 * développement a été extrait dans `dev-server.js` pour garder
 * `home.js` propre et dédié au front.
 */

(function () {
	'use strict';

	const $ = id => document.getElementById(id);

	const fileInput = $('file-input');
	const dropArea = $('drop-area');
	const previewImg = $('preview-img');
	const previewVideo = $('preview-video');
	const previewWrapper = $('preview-wrapper');
	const simulateBtn = $('simulate-btn');
	const resultEl = $('result');
	const spinner = $('spinner');

	if (!fileInput || !dropArea || !previewWrapper || !simulateBtn || !resultEl || !spinner) {
		// Si des éléments manquent, on logge et on arrête l'initialisation.
		// Ceci évite des erreurs si le script est inclus ailleurs.
		/* eslint-disable no-console */
		console.error('home.js: éléments DOM manquants — initialisation annulée.');
		return;
	}

	let currentFile = null;
	let currentObjectUrl = null; // pour les vidéos blob URLs

	function showSpinner(show) {
		spinner.classList.toggle('hidden', !show);
		spinner.setAttribute('aria-hidden', String(!show));
	}

	function setResult(text, isDeepfake = null, score = null) {
		let html = `<p>${text}</p>`;
		if (isDeepfake !== null) {
			html += `<p class="badge ${isDeepfake ? 'deep' : 'real'}">${isDeepfake ? 'Deepfake' : 'Authentique'}</p>`;
		}
		if (score !== null) html += `<p>Score: ${Number(score).toFixed(3)}</p>`;
		resultEl.innerHTML = html;
	}

	function resetResult() {
		resultEl.innerHTML = '';
	}

	// Events
	// Quand l'utilisateur sélectionne un fichier via l'input file,
	// on récupère le premier fichier sélectionné et on appelle
	// `previewFile` pour afficher un aperçu localement.
	fileInput.addEventListener('change', (e) => {
		const f = e.target.files && e.target.files[0];
		if (!f) return;
		previewFile(f);
	});

	// Drag & drop handlers:
	// - `dragover`: empêcher le comportement par défaut pour autoriser le drop
	//   et ajouter une classe visuelle.
	// - `dragleave`: retirer la classe visuelle quand l'élément quitte la zone.
	// - `drop`: empêcher le comportement par défaut, retirer la classe visuelle
	//   et prévisualiser le premier fichier déposé (si présent).
	dropArea.addEventListener('dragover', (e) => {
		e.preventDefault();
		dropArea.classList.add('dragover');
	});
	dropArea.addEventListener('dragleave', () => dropArea.classList.remove('dragover'));
	dropArea.addEventListener('drop', (e) => {
		e.preventDefault();
		dropArea.classList.remove('dragover');
		const f = e.dataTransfer.files && e.dataTransfer.files[0];
		if (f) previewFile(f);
	});

	// Lit le fichier image en local via FileReader et place le résultat
	// dans l'attribut `src` de l'élément <img> pour afficher l'aperçu.
	function clearPreview() {
		if (currentObjectUrl) {
			URL.revokeObjectURL(currentObjectUrl);
			currentObjectUrl = null;
		}
		if (previewImg) {
			previewImg.src = '';
			previewImg.classList.remove('visible');
		}
		if (previewVideo) {
			previewVideo.removeAttribute('src');
			previewVideo.classList.remove('visible');
			previewVideo.load();
		}
		previewWrapper.classList.remove('has-image', 'has-video');
	}

	function previewFile(file) {
		const isImage = file.type.startsWith('image/');
		const isVideo = file.type.startsWith('video/');
		if (!isImage && !isVideo) {
			alert('Veuillez sélectionner une image ou une vidéo.');
			return;
		}
		currentFile = file;
		clearPreview();
		resetResult();
		if (isImage) {
			const reader = new FileReader();
			reader.onload = () => {
				previewImg.src = reader.result;
				previewImg.classList.add('visible');
				previewWrapper.classList.add('has-image');
			};
			reader.readAsDataURL(file);
		} else if (isVideo) {
			currentObjectUrl = URL.createObjectURL(file);
			previewVideo.src = currentObjectUrl;
			previewVideo.classList.add('visible');
			previewWrapper.classList.add('has-video');
			// charger le preview (optionnel: autoplay court)
			previewVideo.load();
		}
	}

	// Gestionnaire du bouton "Simuler résultat" :
	// envoie la vidéo/image au serveur pour obtenir une prédiction.
	simulateBtn.addEventListener('click', () => {
		if (!currentFile) {
			alert('Sélectionnez d\'abord un fichier (image ou vidéo) pour la prédiction.');
			return;
		}
		showSpinner(true);
		setResult('Traitement en cours...');
		
		// Crée un FileReader pour lire le fichier en tant que buffer
		const reader = new FileReader();
		reader.onload = () => {
			const arrayBuffer = reader.result;
			
			// Envoie le fichier au serveur
			fetch('/predict', {
				method: 'POST',
				body: arrayBuffer,
				headers: {
					'Content-Type': currentFile.type || 'application/octet-stream'
				}
			})
			.then(response => response.json())
			.then(data => {
				showSpinner(false);
				if (data.error) {
					setResult(`Erreur: ${data.error}`);
				} else {
					const score = data.score;
					const isDeep = score > 0.5;
					setResult(
						isDeep ? `Résultat: vidéo deepfake probable.` : `Résultat: vidéo probablement authentique.`,
						isDeep,
						score
					);
				}
			})
			.catch(err => {
				showSpinner(false);
				setResult(`Erreur de connexion: ${err.message}`);
			});
		};
		reader.readAsArrayBuffer(currentFile);
	});

})();
