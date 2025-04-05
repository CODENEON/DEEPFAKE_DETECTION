document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const themeToggle = document.getElementById('theme-toggle');
    const modeToggles = document.querySelectorAll('.detection-mode-toggle');
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const uploadFormats = document.getElementById('upload-formats');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const videoPreview = document.getElementById('video-preview');
    const removeFileBtn = document.getElementById('remove-file');
    const detectButton = document.getElementById('detect-button');
    const progressContainer = document.getElementById('upload-progress-container');
    const progressBar = document.getElementById('upload-progress-bar');
    const resultsContainer = document.getElementById('results-container');
    const resultBadge = document.getElementById('result-badge');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceValue = document.getElementById('confidence-value');
    const ethicalScoreContainer = document.getElementById('ethical-score-container');
    const ethicalScoreCenter = document.getElementById('ethical-score-center');
    const ethicalText = document.getElementById('ethical-text');
    const errorAlert = document.getElementById('error-alert');
    const recentDetectionsList = document.getElementById('recent-detections-list');
    
    // Feedback elements
    const userFeedbackSection = document.getElementById('user-feedback-section');
    const reasonSelect = document.getElementById('reason-select');
    const ethicalScoreInput = document.getElementById('ethical-score-input');
    const ethicalScoreValue = document.getElementById('ethical-score-value');
    const submitFeedbackBtn = document.getElementById('submit-feedback-btn');
    const feedbackSuccess = document.getElementById('feedback-success');

    // State Variables
    let currentMode = 'image';
    let currentFile = null;
    let ethicalChart = null;
    let recentDetections = [];
    let currentDetectionData = null;

    // Theme Management
    function setTheme(isDark) {
        document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
        localStorage.setItem('darkTheme', isDark ? 'true' : 'false');
        
        // Update toggle icon
        themeToggle.innerHTML = isDark ? 
            '<i class="fas fa-sun"></i>' : 
            '<i class="fas fa-moon"></i>';
    }

    // Initialize theme from localStorage or default to light theme
    function initializeTheme() {
        const savedTheme = localStorage.getItem('darkTheme');
        if (savedTheme !== null) {
            setTheme(savedTheme === 'true');
        } else {
            // Default to light theme
            setTheme(false);
        }
    }

    // Toggle theme when button is clicked
    themeToggle.addEventListener('click', function() {
        const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
        setTheme(currentTheme === 'light');
    });

    // Initialize theme on page load
    initializeTheme();

    // Toggle between image and video modes
    modeToggles.forEach(toggle => {
        toggle.addEventListener('change', function() {
            currentMode = this.value;
            
            // Update accepted file formats
            if (currentMode === 'image') {
                fileInput.setAttribute('accept', '.jpg,.jpeg,.png');
                uploadFormats.textContent = 'Accepted formats: .jpg, .jpeg, .png';
            } else {
                fileInput.setAttribute('accept', '.mp4,.mov,.avi');
                uploadFormats.textContent = 'Accepted formats: .mp4, .mov, .avi';
            }
            
            // Reset file input and previews
            resetFileInput();
        });
    });

    // Drag and Drop Functionality
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function() {
        this.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        this.classList.remove('dragover');
        
        if (e.dataTransfer.files.length) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    // Click to upload
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        if (this.files.length) {
            handleFileSelection(this.files[0]);
        }
    });

    // File handling
    function handleFileSelection(file) {
        // Validate file type based on current mode
        const validFile = validateFile(file);
        
        if (!validFile) {
            showError(`Invalid file type. Please upload ${currentMode === 'image' ? 'JPG, JPEG, or PNG' : 'MP4, MOV, or AVI'}.`);
            return;
        }
        
        currentFile = file;
        showFilePreview(file);
        detectButton.disabled = false;
    }

    function validateFile(file) {
        const fileName = file.name.toLowerCase();
        
        if (currentMode === 'image') {
            return fileName.endsWith('.jpg') || fileName.endsWith('.jpeg') || fileName.endsWith('.png');
        } else {
            return fileName.endsWith('.mp4') || fileName.endsWith('.mov') || fileName.endsWith('.avi');
        }
    }

    function showFilePreview(file) {
        // Create object URL for preview
        const objectUrl = URL.createObjectURL(file);
        
        // Show preview based on file type
        if (currentMode === 'image') {
            imagePreview.src = objectUrl;
            imagePreview.style.display = 'block';
            videoPreview.style.display = 'none';
            videoPreview.src = '';
        } else {
            videoPreview.src = objectUrl;
            videoPreview.style.display = 'block';
            imagePreview.style.display = 'none';
            imagePreview.src = '';
        }
        
        // Show preview container
        previewContainer.style.display = 'block';
        
        // Hide results if they were shown
        resultsContainer.style.display = 'none';
        
        // Hide feedback success if it was shown
        feedbackSuccess.style.display = 'none';
    }

    // Remove file button
    removeFileBtn.addEventListener('click', function() {
        resetFileInput();
    });

    function resetFileInput() {
        // Clear file input
        fileInput.value = '';
        currentFile = null;
        
        // Hide preview
        previewContainer.style.display = 'none';
        imagePreview.src = '';
        videoPreview.src = '';
        
        // Disable detect button
        detectButton.disabled = true;
        
        // Hide results
        resultsContainer.style.display = 'none';
        
        // Hide error alert
        errorAlert.style.display = 'none';
        
        // Hide feedback form
        userFeedbackSection.style.display = 'none';
        
        // Reset current detection data
        currentDetectionData = null;
    }

    // Load reasons for deepfakes on page load
    fetchDeepfakeReasons();

    function fetchDeepfakeReasons() {
        fetch('/api/deepfake-reasons')
            .then(response => response.json())
            .then(data => {
                // Clear existing options
                reasonSelect.innerHTML = '<option value="" selected disabled>Select a reason...</option>';
                
                // Add each reason as an option
                data.forEach(reason => {
                    const option = document.createElement('option');
                    option.value = reason.id;
                    option.textContent = reason.text;
                    reasonSelect.appendChild(option);
                });
            })
            .catch(error => {
                console.error('Error fetching deepfake reasons:', error);
                showError('Failed to load deepfake reasons. Please refresh the page.');
            });
    }

    // Detect button
    detectButton.addEventListener('click', function() {
        if (!currentFile) {
            showError('Please select a file first.');
            return;
        }
        
        // Show progress bar
        progressContainer.style.display = 'block';
        progressBar.style.width = '0%';
        
        // Create form data
        const formData = new FormData();
        formData.append('file', currentFile);
        
        // Determine endpoint based on current mode
        const endpoint = currentMode === 'image' ? '/detect/image' : '/detect/video';
        
        // Simulate progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += 5;
            if (progress <= 90) {
                progressBar.style.width = `${progress}%`;
            }
        }, 100);
        
        // Send request to backend
        fetch(endpoint, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            clearInterval(progressInterval);
            progressBar.style.width = '100%';
            
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.error || 'Detection failed. Please try again.');
                });
            }
            
            return response.json();
        })
        .then(data => {
            // Store the detection data
            currentDetectionData = data;
            
            // Hide progress after a small delay
            setTimeout(() => {
                progressContainer.style.display = 'none';
                
                // Display results
                displayResults(data);
                
                // Add to recent detections
                addToRecentDetections(data);
                
                // Show feedback form if it's a fake
                if (data.result.toLowerCase() === 'fake') {
                    userFeedbackSection.style.display = 'block';
                    feedbackSuccess.style.display = 'none';
                } else {
                    userFeedbackSection.style.display = 'none';
                }
            }, 500);
        })
        .catch(error => {
            clearInterval(progressInterval);
            progressContainer.style.display = 'none';
            showError(error.message);
        });
    });

    // Display results
    function displayResults(data) {
        // Set result badge
        resultBadge.textContent = data.result.toUpperCase();
        resultBadge.className = `badge ${data.result.toLowerCase()} animated-fade-in`;
        
        // Set confidence score
        const confidencePercent = data.confidence.toFixed(1);
        confidenceBar.style.width = `${confidencePercent}%`;
        confidenceValue.textContent = `${confidencePercent}%`;
        
        // Set color based on result
        confidenceBar.className = `progress-bar bg-${data.result === 'real' ? 'success' : 'danger'}`;
        
        // Handle ethical score (only for fake media)
        if (data.result === 'fake' && data.ethical_score !== undefined) {
            ethicalScoreContainer.style.display = 'block';
            ethicalScoreCenter.textContent = Math.round(data.ethical_score);
            
            // Update ethical chart
            updateEthicalChart(data.ethical_score);
            
            // Set ethical impact text
            if (data.ethical_score < 30) {
                ethicalText.textContent = 'Low concern - Minor manipulation with limited potential harm.';
            } else if (data.ethical_score < 70) {
                ethicalText.textContent = 'Moderate concern - Significant manipulation with moderate ethical impact.';
            } else {
                ethicalText.textContent = 'High concern - Severe manipulation with significant potential for harm.';
            }
        } else {
            ethicalScoreContainer.style.display = 'none';
        }
        
        // Show results container
        resultsContainer.style.display = 'block';
    }

    // Update ethical chart
    function updateEthicalChart(score) {
        if (ethicalChart) {
            ethicalChart.destroy();
        }
        
        const ctx = document.getElementById('ethical-chart').getContext('2d');
        
        // Determine color based on ethical score
        let color;
        if (score < 30) {
            color = '#34a853';  // Green for low concern
        } else if (score < 70) {
            color = '#fbbc05';  // Yellow for moderate concern
        } else {
            color = '#ea4335';  // Red for high concern
        }
        
        ethicalChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [score, 100 - score],
                    backgroundColor: [color, '#e0e0e0'],
                    borderWidth: 0
                }]
            },
            options: {
                cutout: '75%',
                responsive: true,
                maintainAspectRatio: true,
                animation: {
                    animateRotate: true,
                    animateScale: true
                },
                plugins: {
                    tooltip: {
                        enabled: false
                    },
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    // Add to recent detections
    function addToRecentDetections(data) {
        // Create a new detection record
        const detection = {
            id: data.detection_id || Date.now(), // Use provided ID or fallback to timestamp
            type: currentMode,
            result: data.result,
            confidence: data.confidence,
            file_type: data.file_type || currentMode,
            timestamp: new Date().toLocaleTimeString()
        };
        
        // Add to the beginning of the array
        recentDetections.unshift(detection);
        
        // Limit to 5 items
        if (recentDetections.length > 5) {
            recentDetections.pop();
        }
        
        // Update UI
        updateRecentDetectionsList();
    }

    function updateRecentDetectionsList() {
        // Clear the list
        recentDetectionsList.innerHTML = '';
        
        // If no detections yet
        if (recentDetections.length === 0) {
            recentDetectionsList.innerHTML = `
                <div class="text-center text-muted">
                    No detections yet. Your detection history will appear here.
                </div>
            `;
            return;
        }
        
        // Add each detection
        recentDetections.forEach((detection, index) => {
            const detectionItem = document.createElement('div');
            const resultClass = detection.result.toLowerCase();
            
            // Create the item with the result class for styling
            detectionItem.className = `recent-item ${resultClass}`;
            // Add animation delay based on item index
            detectionItem.style.animationDelay = `${index * 0.1}s`;
            detectionItem.classList.add('animated-fade-in');
            
            detectionItem.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <div class="recent-item-title">
                        <i class="fas fa-${detection.type === 'image' ? 'image' : 'video'}"></i>
                        ${detection.type.charAt(0).toUpperCase() + detection.type.slice(1)} Analysis
                    </div>
                    <span class="recent-item-result ${resultClass}">
                        ${detection.result.toUpperCase()}
                    </span>
                </div>
                <div class="recent-item-confidence">
                    Confidence: <strong>${detection.confidence.toFixed(1)}%</strong>
                </div>
                <div class="recent-item-time">
                    <i class="far fa-clock"></i> ${detection.timestamp}
                </div>
            `;
            recentDetectionsList.appendChild(detectionItem);
        });
    }

    // Error handling
    function showError(message) {
        errorAlert.textContent = message;
        errorAlert.style.display = 'block';
        
        // Hide error after 5 seconds
        setTimeout(() => {
            errorAlert.style.display = 'none';
        }, 5000);
    }
    
    // Ethical score input handling
    ethicalScoreInput.addEventListener('input', function() {
        ethicalScoreValue.textContent = this.value;
    });
    
    // Submit feedback
    submitFeedbackBtn.addEventListener('click', function() {
        // Validate form
        if (!reasonSelect.value) {
            showError('Please select a reason for why you think this is a deepfake.');
            return;
        }
        
        if (!currentDetectionData) {
            showError('Detection data not found. Please try again.');
            return;
        }
        
        // Prepare feedback data
        const feedbackData = {
            detection_id: currentDetectionData.detection_id,
            reason_id: parseInt(reasonSelect.value),
            ethical_score: parseInt(ethicalScoreInput.value),
            file_type: currentDetectionData.file_type,
            confidence_score: currentDetectionData.confidence
        };
        
        // Send feedback to server
        fetch('/api/submit-feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(feedbackData)
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.error || 'Failed to submit feedback. Please try again.');
                });
            }
            return response.json();
        })
        .then(data => {
            // Show success message
            feedbackSuccess.style.display = 'block';
            
            // Reset form
            reasonSelect.selectedIndex = 0;
            ethicalScoreInput.value = 5;
            ethicalScoreValue.textContent = '5';
            
            // Hide form after 3 seconds
            setTimeout(() => {
                userFeedbackSection.style.display = 'none';
            }, 3000);
        })
        .catch(error => {
            showError(error.message);
        });
    });
});
