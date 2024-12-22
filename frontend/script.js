const submitBtn = document.getElementById('submitBtn');
const resultDiv = document.getElementById('result');
const imageInput = document.getElementById('imageInput');

submitBtn.addEventListener('click', async () => {
    // Check if a file is selected
    if (imageInput.files.length === 0) {
        resultDiv.innerHTML = 'Please select an image to upload.';
        return;
    }

    const formData = new FormData();
    formData.append('image', imageInput.files[0]);

    try {
        // Send the image to the backend for prediction
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            resultDiv.innerHTML = `Prediction: ${data.prediction}`;
        } else {
            resultDiv.innerHTML = 'Error: Could not classify the image.';
        }
    } catch (error) {
        resultDiv.innerHTML = 'Error: Failed to communicate with the server.';
        console.error('Error:', error);
    }
});
