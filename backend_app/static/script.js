document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const predictBtn = document.getElementById('predictBtn');
    const imagePreview = document.getElementById('imagePreview');
    const resultDiv = document.getElementById('result');
    const loader = document.getElementById('loader');
    let file = null;

    imageUpload.addEventListener('change', (event) => {
        file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.innerHTML = `<img src="${e.target.result}" alt="Image preview"/>`;
                predictBtn.disabled = false;
                resultDiv.innerText = '';
            };
            reader.readAsDataURL(file);
        }
    });

    predictBtn.addEventListener('click', async () => {
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        loader.style.display = 'block';
        resultDiv.innerText = '';
        predictBtn.disabled = true;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            if (data.error) {
                resultDiv.innerText = `Error: ${data.error}`;
            } else {
                resultDiv.innerText = `Predicted Emotion: ${data.emotion} (${data.confidence})`;
            }

        } catch (error) {
            console.error('Error:', error);
            resultDiv.innerText = 'An error occurred. Please try again.';
        } finally {
            loader.style.display = 'none';
            predictBtn.disabled = false;
        }
    });
});