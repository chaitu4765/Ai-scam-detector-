document.addEventListener('DOMContentLoaded', () => {
    // State
    let activeTab = 'text';
    let loading = false;
    const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
        ? 'http://localhost:8000'
        : '/api';

    // DOM Elements
    const tabBtns = document.querySelectorAll('.tab-btn');
    const textInputArea = document.getElementById('text-input-area');
    const qrInputArea = document.getElementById('qr-input-area');
    const textArea = document.getElementById('input-text');
    const qrFile = document.getElementById('qr-file');
    const scanBtn = document.getElementById('scan-btn');
    const btnText = scanBtn.querySelector('.btn-text');
    const btnIcon = scanBtn.querySelector('.btn-icon');
    const loadingSpinner = document.getElementById('loading-spinner');

    const resultContainer = document.getElementById('result-container');
    const resultCard = document.getElementById('result-card');
    const resultStatus = document.getElementById('result-status');
    const resultMessage = document.getElementById('result-message');
    const resultConfidence = document.getElementById('result-confidence');
    const iconWarning = document.getElementById('icon-warning');
    const iconSafe = document.getElementById('icon-safe');

    // Tab Switching
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            if (loading) return;

            const tab = btn.getAttribute('data-tab');
            activeTab = tab;

            // Update UI
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            if (tab === 'qr') {
                textInputArea.classList.add('hidden');
                qrInputArea.classList.remove('hidden');
            } else {
                textInputArea.classList.remove('hidden');
                qrInputArea.classList.add('hidden');
                textArea.placeholder = tab === 'url' ? "Paste suspicious URL here..." : "Paste email content here to scan for threats...";
            }
        });
    });

    // Handle Scan
    scanBtn.addEventListener('click', async () => {
        if (loading) return;

        const content = textArea.value.trim();
        const file = qrFile.files[0];

        if (activeTab !== 'qr' && !content) return;
        if (activeTab === 'qr' && !file) {
            alert("Please select a QR code image.");
            return;
        }

        setLoading(true);
        resultContainer.classList.add('hidden');

        try {
            let response;
            if (activeTab === 'qr') {
                const formData = new FormData();
                formData.append('file', file);
                response = await fetch(`/api/decode/qr`, {
                    method: 'POST',
                    body: formData
                });
            } else {
                // ✅ FINAL ABSOLUTE URL FETCH (v10.0)
                const backendUrl = window.location.origin + "/api/scan";
                console.log("Connecting to:", backendUrl);

                response = await fetch(backendUrl, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    body: JSON.stringify({ input: content })
                });
            }

            if (!response.ok) {
                throw new Error("Server error " + response.status);
            }

            const data = await response.json();

            setTimeout(() => {
                showResult(data);
                setLoading(false);
            }, 800);

        } catch (error) {
            console.error("Fetch failure", error);
            setLoading(false);

            // Fallback result with clean error message (no double prefix)
            showResult({
                safe: true,
                phishing: false,
                confidence: 65,
                is_error: true,
                error_msg: "Backend unreachable"
            });
        }
    });

    function setLoading(val) {
        loading = val;
        scanBtn.disabled = val;
        if (val) {
            btnText.classList.add('hidden');
            btnIcon.classList.add('hidden');
            loadingSpinner.classList.remove('hidden');
        } else {
            btnText.classList.remove('hidden');
            btnIcon.classList.remove('hidden');
            loadingSpinner.classList.add('hidden');
        }
    }

    function showResult(data) {
        resultContainer.classList.remove('hidden');

        const isSafe = data.safe;
        const confidenceValue = data.confidence;

        resultCard.classList.remove('danger-theme', 'safe-theme');
        iconWarning.classList.add('hidden');
        iconSafe.classList.add('hidden');

        if (isSafe) {
            resultCard.classList.add('safe-theme');
            resultStatus.textContent = 'SAFE';
            resultMessage.textContent = activeTab === 'url' ? 'URL appears safe.' : 'Message appears safe.';
            iconSafe.classList.remove('hidden');
        } else {
            resultCard.classList.add('danger-theme');
            resultStatus.textContent = 'DANGEROUS';
            resultMessage.textContent = 'Suspicious content detected.';
            iconWarning.classList.remove('hidden');
        }

        if (data.is_error) {
            // Clean display: "Connection error: Backend unreachable"
            resultMessage.textContent = `Connection error: ${data.error_msg}`;
        }

        resultConfidence.textContent = `${confidenceValue}%`;
        lucide.createIcons();
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
});
