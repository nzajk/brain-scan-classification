const imageInput = document.getElementById('imageInput');
const previewImage = document.getElementById('previewImage');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const uploadZone = document.getElementById('uploadZone');
const fileInfo = document.getElementById('fileInfo');
const fileInfoName = document.getElementById('fileInfoName');
const clearBtn = document.getElementById('clearBtn');
const submitBtn = document.getElementById('submitBtn');
const btnText = document.getElementById('btnText');
const spinner = document.getElementById('spinner');
const result = document.getElementById('result');
const resultIndicator = document.getElementById('resultIndicator');
const resultTitle = document.getElementById('resultTitle');
const resultNote = document.getElementById('resultNote');
const confValue = document.getElementById('confValue');
const confFill = document.getElementById('confFill');
const timeValue = document.getElementById('timeValue');
const errorMsg = document.getElementById('errorMsg');

function loadFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const reader = new FileReader();
  reader.onload = e => {
    previewImage.src = e.target.result;
    previewImage.style.display = 'block';
    uploadPlaceholder.style.display = 'none';
    uploadZone.classList.add('has-image');
    fileInfoName.textContent = file.name;
    fileInfo.style.display = 'flex';
  };
  reader.readAsDataURL(file);
  submitBtn.disabled = false;
  btnText.textContent = 'Classify Scan';
  result.style.display = 'none';
  errorMsg.style.display = 'none';
  confFill.style.width = '0%';
}

function clearFile() {
  imageInput.value = '';
  previewImage.style.display = 'none';
  previewImage.src = '';
  uploadPlaceholder.style.display = 'flex';
  uploadZone.classList.remove('has-image');
  fileInfo.style.display = 'none';
  submitBtn.disabled = true;
  btnText.textContent = 'Select a scan to begin';
  result.style.display = 'none';
  errorMsg.style.display = 'none';
}

function showResult(prediction, confidence_score, elapsed) {
  const isNormal = prediction.toLowerCase().includes('no tumor');

  result.style.display = 'block';
  resultIndicator.className = 'result-indicator ' + (isNormal ? 'normal' : 'abnormal');
  resultTitle.textContent = prediction;
  resultNote.textContent = isNormal
    ? 'Scan appears within normal parameters.'
    : 'Potential irregularity found — consult a specialist.';
  timeValue.textContent = elapsed;

  const confidence = Math.max(0, Math.round(parseFloat(confidence_score) * 100));

  confValue.textContent = `${confidence}%`;
  confFill.style.width = `${confidence}%`;
}

function showError(message) {
  result.style.display = 'block';
  resultIndicator.className = 'result-indicator abnormal';
  resultTitle.textContent = 'Error';
  resultNote.textContent = message;
  confValue.textContent = '—';
  timeValue.textContent = '—';
}

clearBtn.addEventListener('click', e => { e.stopPropagation(); clearFile(); });
imageInput.addEventListener('change', e => loadFile(e.target.files[0]));

uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  loadFile(e.dataTransfer.files[0]);
});

submitBtn.addEventListener('click', async () => {
  if (!imageInput.files[0]) {
    errorMsg.style.display = 'block';
    return;
  }

  submitBtn.disabled = true;
  spinner.style.display = 'block';
  btnText.textContent = 'Analyzing...';
  result.style.display = 'none';
  errorMsg.style.display = 'none';
  const start = Date.now();

  const formData = new FormData();
  formData.append('image', imageInput.files[0]);

  try {
    const response = await fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      body: formData
    });

    const elapsed = ((Date.now() - start) / 1000).toFixed(2) + 's';

    if (response.ok) {
      const data = await response.json();
      showResult(data.prediction, data.confidence_score, elapsed);
    } else {
      showError('Could not classify the image. Server returned an error.');
    }
  } catch (error) {
    const elapsed = ((Date.now() - start) / 1000).toFixed(2) + 's';
    console.error('Error:', error);
    showError('Failed to communicate with the server. Is it running?');
    timeValue.textContent = elapsed;
  } finally {
    spinner.style.display = 'none';
    submitBtn.disabled = false;
    btnText.textContent = 'Reclassify';
  }
});