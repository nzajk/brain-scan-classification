const submitBtn = document.getElementById('submitBtn');
        const resultDiv = document.getElementById('result');
        const imageInput = document.getElementById('imageInput');

        submitBtn.addEventListener('click', async () => {
            const formData = new FormData();
            formData.append('image', imageInput.files[0]);

            try {
                // send the image to the backend for prediction
                const response = await fetch('/predict', {
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