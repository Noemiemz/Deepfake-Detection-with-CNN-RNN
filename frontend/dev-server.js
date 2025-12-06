// Petit serveur de développement statique
// Usage: `node dev-server.js` (optionnel: définir PORT env var)

const http = require('http');
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');

const port = process.env.PORT || 8000;
// Se base sur le répertoire courant (frontend)
const root = __dirname;

const mime = {
  '.html': 'text/html', '.js': 'application/javascript', '.css': 'text/css',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.svg': 'image/svg+xml',
  '.json': 'application/json', '.txt': 'text/plain'
};

const server = http.createServer((req, res) => {
  // Ajouter les headers CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  // Gérer les requêtes OPTIONS
  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }

  try {
    // Endpoint pour les prédictions
    if (req.method === 'POST' && req.url === '/predict') {
      let body = [];
      req.on('data', chunk => {
        body.push(chunk);
      });
      req.on('end', () => {
        try {
          const buffer = Buffer.concat(body);
          // Sauvegarde le fichier temporaire dans le répertoire parent
          const tempPath = path.join(root, '..', 'temp_video.mp4');
          fs.writeFile(tempPath, buffer, (writeErr) => {
            if (writeErr) {
              console.error('Erreur sauvegarde:', writeErr);
              res.statusCode = 500;
              res.setHeader('Content-Type', 'application/json');
              res.end(JSON.stringify({ error: 'Erreur lors de la sauvegarde du fichier' }));
              return;
            }
            
            console.log('Fichier sauvegardé, exécution du script Python...');
            
            // Appelle le script Python (depuis le répertoire parent)
            const pythonScript = path.join(root, '..', 'src', 'predict_video.py');
            const modelPath = path.join(root, '..', 'deepfake_detector.pth');
            const { spawn } = require('child_process');
            const python = spawn('python', [pythonScript, '--video_path', tempPath, '--model_path', modelPath, '--device', 'cpu']);
            
            let pythonOutput = '';
            let pythonError = '';
            
            python.stdout.on('data', (data) => {
              pythonOutput += data.toString();
              console.log('Python stdout:', data.toString());
            });
            
            python.stderr.on('data', (data) => {
              pythonError += data.toString();
              console.log('Python stderr:', data.toString());
            });
            
            python.on('error', (err) => {
              console.error('Erreur spawn Python:', err);
              res.statusCode = 500;
              res.setHeader('Content-Type', 'application/json');
              res.end(JSON.stringify({ error: 'Impossible de lancer le script Python', details: err.message }));
              fs.unlink(tempPath, () => {});
            });
            
            python.on('close', (code) => {
              // Supprime le fichier temporaire
              fs.unlink(tempPath, () => {});
              
              if (code !== 0) {
                console.error('Python exit code:', code);
                res.statusCode = 500;
                res.setHeader('Content-Type', 'application/json');
                res.end(JSON.stringify({ error: 'Erreur lors de la prédiction', details: pythonError }));
                return;
              }
              
              // Parse le résultat (cherche le score)
              const scoreMatch = pythonOutput.match(/Score: ([\d.]+)/);
              const score = scoreMatch ? parseFloat(scoreMatch[1]) : null;
              
              console.log('Prédiction réussie, score:', score);
              res.statusCode = 200;
              res.setHeader('Content-Type', 'application/json');
              res.end(JSON.stringify({ score, output: pythonOutput }));
            });
          });
        } catch (err) {
          console.error('Erreur dans le traitement POST:', err);
          res.statusCode = 500;
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ error: 'Erreur serveur', details: err.message }));
        }
      });
      return;
    }
    
    // Fichiers statiques
    let urlPath = decodeURIComponent(req.url.split('?')[0]);
    if (urlPath === '/') urlPath = '/index.html';
    const filePath = path.join(root, urlPath);
    fs.stat(filePath, (err, stats) => {
      if (err || !stats.isFile()) {
        res.statusCode = 404;
        res.setHeader('Content-Type', 'text/plain; charset=utf-8');
        res.end('404 - Not Found');
        return;
      }
      const ext = path.extname(filePath).toLowerCase();
      res.statusCode = 200;
      res.setHeader('Content-Type', mime[ext] || 'application/octet-stream');
      const stream = fs.createReadStream(filePath);
      stream.pipe(res);
    });
  } catch (e) {
    res.statusCode = 500;
    res.end('Server error');
  }
});

server.listen(port, () => {
  const url = `http://localhost:${port}/index.html`;
  console.log(`Serving ${root} at ${url}`);
  // Ouvre le navigateur par défaut (Windows/macOS/Linux)
  const plat = process.platform;
  let cmd;
  if (plat === 'win32') cmd = `start "" "${url}"`;
  else if (plat === 'darwin') cmd = `open "${url}"`;
  else cmd = `xdg-open "${url}"`;
  exec(cmd, (err) => {
    if (err) console.log('Impossible d\'ouvrir automatiquement le navigateur:', err.message);
  });
});
