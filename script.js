document.getElementById('analyzeBtn').addEventListener('click', uploadImage);
document.getElementById('imageInput').addEventListener('change', showPreview);

function showPreview() {
  const file = this.files[0];
  const preview = document.getElementById('imagePreview');

  if (file) {
    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';
  } else {
    preview.style.display = 'none';
  }
}

function uploadImage() {
  const fileInput = document.getElementById('imageInput');
  const file = fileInput.files[0];
  const result = document.getElementById('result');
  const error = document.getElementById('error');

  if (!file) {
    error.textContent = '⚠ Please select an image before analyzing.';
    error.style.display = 'block';
    result.style.display = 'none';
    return;
  }

  const formData = new FormData();
  formData.append('image', file);

  fetch('http://127.0.0.1:5000/predict', {
    method: 'POST',
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      if (data.status !== 200) {
        throw new Error(data.error || 'Unexpected error occurred');
      }

      // Show predicted disease name and confidence
      document.getElementById('disease').textContent = `${data.label} (Confidence: ${data.score.toFixed(2)}%)`;

      // Show disease explanation
      document.getElementById('wikiInfo').textContent = data.info || 'No additional information available.';

      result.style.display = 'block';
      error.style.display = 'none';
    })
    .catch(err => {
      error.textContent = `❌ ${err.message}`;
      error.style.display = 'block';
      result.style.display = 'none';
    });
}
