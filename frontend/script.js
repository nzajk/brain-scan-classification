const submitBtn = document.getElementById('submitBtn');
const resultDiv = document.getElementById('result');
const imageInput = document.getElementById('imageInput');
const previewImage = document.getElementById('previewImage');

// event listener for when a file is selected
imageInput.addEventListener('change', (event) => {
    const file = event.target.files[0];

    if (file) {
        const reader = new FileReader();
        
        // when the file is loaded, set the preview image
        reader.onload = () => {
            previewImage.src = reader.result;
            previewImage.classList.remove('hidden');
        };

        reader.readAsDataURL(file);
    } else {
        previewImage.classList.add('hidden');
    }
});

submitBtn.addEventListener('click', async () => {
    // check if a file is selected
    if (imageInput.files.length === 0) {
        resultDiv.innerHTML = 'Please select an image to upload.';
        return;
    }

    const formData = new FormData();
    formData.append('image', imageInput.files[0]);

    try {
        // send the image to the backend for prediction
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            resultDiv.innerHTML = `Prediction: ${data.prediction}`;
            console.log(data);
        } else {
            resultDiv.innerHTML = 'Error: Could not classify the image.';
        }
    } catch (error) {
        resultDiv.innerHTML = 'Error: Failed to communicate with the server.';
        console.error('Error:', error);
    }
});
