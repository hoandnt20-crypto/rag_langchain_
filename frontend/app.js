// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const searchModeSelect = document.getElementById('searchModeSelect');
const textSearchContent = document.getElementById('textSearchContent');
const imageSearchContent = document.getElementById('imageSearchContent');
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const topkInput = document.getElementById('topkInput');
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');

// Image Preview Elements
const previewImage = document.getElementById('previewImage');
const uploadDefaultContent = document.getElementById('uploadDefaultContent');
const clearImageBtn = document.getElementById('clearImageBtn');

const uploadBtn = document.getElementById('uploadBtn');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('resultsSection');
const resultsCount = document.getElementById('resultsCount');
const videoResults = document.getElementById('videoResults');
const errorMessage = document.getElementById('errorMessage');
const emptyState = document.getElementById('emptyState');

// Video Preview Elements
const videoPreview = document.getElementById('videoPreview');
const videoTitle = document.getElementById('videoTitle');
const videoTime = document.getElementById('videoTime');
const closePreviewBtn = document.getElementById('closePreview');

// === YOUTUBE API SETUP ===
let player;
let currentMode = 'text';
let selectedImage = null;

// Khởi tạo Player khi API sẵn sàng (theo cách bạn yêu cầu)
// Lưu ý: Đảm bảo script YouTube API đã được load trong HTML
window.onYouTubeIframeAPIReady = function () {
    console.log("YouTube IFrame API script loaded."); // Debug log
    // Tạo player gắn vào thẻ div có id="youtubePlayer"
    player = new YT.Player('youtubePlayer', {
        height: '200',
        width: '100%',
        videoId: '', // Chưa load video nào
        playerVars: {
            'playsinline': 1,
            'autoplay': 1, // Tự động phát khi load video mới
            'modestbranding': 1,
            'rel': 0,
            // Quan trọng: Thêm origin để tránh lỗi 153 trong một số môi trường
            'origin': window.location.origin
        },
        events: {
            'onStateChange': onPlayerStateChange
        }
    });
};

function onPlayerStateChange(event) {
    // Có thể thêm logic xử lý khi video kết thúc hoặc tạm dừng
}

// === CÁC CHỨC NĂNG CHÍNH ===

// Chuyển đổi chế độ tìm kiếm
searchModeSelect.addEventListener('change', (e) => switchMode(e.target.value));

function switchMode(mode) {
    currentMode = mode;
    if (mode === 'text' || mode === "description" || mode === "transcription") {
        textSearchContent.classList.add('active');
        imageSearchContent.classList.remove('active');
    } else {
        imageSearchContent.classList.add('active');
        textSearchContent.classList.remove('active');
    }
    hideResults();
    hideError();
    clearPreview();
}

// Xử lý tìm kiếm thống nhất (Text hoặc Image)
searchBtn.addEventListener('click', performUnifiedSearch);

// Tự động mở rộng textarea và xử lý phím Enter
searchInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        performUnifiedSearch();
    }
});

// Hàm tìm kiếm thống nhất - kiểm tra mode và gọi hàm tương ứng
async function performUnifiedSearch() {
    if (currentMode === 'text' || currentMode === 'description' || currentMode === 'transcription') {
        await performTextSearch();
    } else {
        await performImageSearch();
    }
}

async function performTextSearch() {
    const query = searchInput.value.trim();
    if (!query) {
        showError('Vui lòng nhập nội dung tìm kiếm');
        return;
    }

    hideError();
    showLoading();
    hideResults();

    try {
        const topK = topkInput.value || 100;
        const formData = new FormData();
        formData.append('query', query);
        formData.append('top_k', topK);

        // Chọn endpoint dựa trên currentMode
        let endpoint;
        if (currentMode === 'description') {
            endpoint = `${API_BASE_URL}/search/description`;
        } else if (currentMode === 'transcription') {
            endpoint = `${API_BASE_URL}/search/transcription`;
        } else {
            endpoint = `${API_BASE_URL}/search/clip_text`;
        }

        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error(`Lỗi HTTP: ${response.status}`);

        const results = await response.json();
        displayResults(results);
    } catch (error) {
        console.error('Lỗi tìm kiếm:', error);
        showError(`Tìm kiếm thất bại: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// === XỬ LÝ ẢNH (UPLOAD & PREVIEW) ===

function clearPreview() {
    selectedImage = null;
    imageInput.value = '';
    previewImage.src = '';
    previewImage.style.display = 'none';
    uploadDefaultContent.style.display = 'flex';
    clearImageBtn.style.display = 'none';
    uploadArea.classList.remove('has-image');
}

function handleImageFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('Vui lòng chọn file ảnh hợp lệ');
        clearPreview();
        return;
    }

    selectedImage = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadDefaultContent.style.display = 'none';
        previewImage.style.display = 'block';
        clearImageBtn.style.display = 'block';
        uploadArea.classList.add('has-image');
    };
    reader.readAsDataURL(file);
    hideError();
}

clearImageBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearPreview();
});

// Drag & Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) handleImageFile(e.dataTransfer.files[0]);
});

imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleImageFile(e.target.files[0]);
});

// Xử lý tìm kiếm Ảnh (được gọi từ performUnifiedSearch)

async function performImageSearch() {
    if (!selectedImage) {
        showError('Vui lòng chọn ảnh trước');
        return;
    }

    hideError();
    showLoading();
    hideResults();

    try {
        const topK = topkInput.value || 100;
        const formData = new FormData();
        formData.append('file', selectedImage);
        formData.append('top_k', topK);

        const response = await fetch(`${API_BASE_URL}/search/clip_image`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error(`Lỗi HTTP: ${response.status}`);

        const results = await response.json();
        displayResults(results);
    } catch (error) {
        console.error('Lỗi tìm kiếm:', error);
        showError(`Tìm kiếm thất bại: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// === HIỂN THỊ KẾT QUẢ & VIDEO ===

function displayResults(results) {
    if (!results || results.length === 0) {
        showError('Không tìm thấy kết quả nào');
        return;
    }

    const totalKeyframes = results.reduce((sum, video) => sum + video.keyframes.length, 0);
    resultsCount.textContent = `Found total ${totalKeyframes} keyframes in ${results.length} videos`;

    videoResults.innerHTML = '';
    results.forEach((video, index) => {
        videoResults.appendChild(createVideoCard(video, index));
    });

    emptyState.style.display = 'none';
    resultsSection.classList.add('active');
}

function createVideoCard(video, index) {
    const card = document.createElement('div');
    card.className = 'video-card';

    const header = document.createElement('div');
    header.className = 'video-header';
    const videoName = document.createElement('h3');
    videoName.className = 'video-name';
    videoName.textContent = video.video_name;
    header.appendChild(videoName);
    card.appendChild(header);

    const keyframesGrid = document.createElement('div');
    keyframesGrid.className = 'keyframes-grid';

    video.keyframes.forEach((Keyframe, idx) => {
        keyframesGrid.appendChild(createKeyframeItem(
            Keyframe.path,
            Keyframe.similarity,
            Keyframe.frame_idx,
            Keyframe.pts_time,
            video.watch_url,
            `${video.video_name}: ${video.title}`
        ));
    });

    card.appendChild(keyframesGrid);
    return card;
}

function createKeyframeItem(keyframePath, similarity, frameIdx, ptsTime, watchUrl, videoName) {
    const item = document.createElement('div');
    item.className = 'keyframe-item';

    const img = document.createElement('img');
    img.className = 'keyframe-img';
    img.src = `${API_BASE_URL}/static/${keyframePath}`;
    img.alt = `Frame ${frameIdx}`;
    img.onerror = () => {
        img.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="150"%3E%3Crect fill="%23333" width="200" height="150"/%3E%3Ctext fill="%23999" x="50%25" y="50%25" text-anchor="middle" dy=".3em"%3EImage not found%3C/text%3E%3C/svg%3E';
    };

    item.appendChild(img);

    item.addEventListener('click', () => {
        playVideo(watchUrl, ptsTime, videoName);
    });

    return item;
}

// === VIDEO PREVIEW LOGIC (SỬA LẠI ĐỂ CHẠY API TRỰC TIẾP) ===

function playVideo(watchUrl, startTime, title) {
    if (!watchUrl) return;

    const videoId = watchUrl;
    if (!videoId) {
        showError('URL video không hợp lệ');
        return;
    }

    // Hiển thị khung video
    videoPreview.classList.add('active');
    videoPreview.scrollIntoView({ behavior: 'smooth', block: 'center' });

    videoTitle.textContent = title;
    videoTime.textContent = `Bắt đầu tại: ${formatTime(startTime)}`;

    // Gọi API trực tiếp, không try-catch
    if (player) {
        player.loadVideoById({
            videoId: videoId,
            startSeconds: Math.floor(startTime)
        });
    } else {
        console.warn('Player chưa sẵn sàng. Đang chờ...');
    }
}

function extractVideoId(url) {
    if (!url) return null;
    const regExp = /^.*((youtu.be\/)|(v\/)|(\/u\/\w\/)|(embed\/)|(watch\?))\??v?=?([^#&?]*).*/;
    const match = url.match(regExp);
    return (match && match[7].length === 11) ? match[7] : null;
}

closePreviewBtn.addEventListener('click', () => {
    // Dừng video khi đóng
    if (player && typeof player.stopVideo === 'function') {
        player.stopVideo();
    }
    videoPreview.classList.remove('active');
});

// UI Helper Functions
function formatTime(seconds) {
    const min = Math.floor(seconds / 60);
    const sec = Math.floor(seconds % 60);
    return `${min}:${sec.toString().padStart(2, '0')}`;
}

function showLoading() {
    loading.classList.add('active');
    searchBtn.disabled = true;
}

function hideLoading() {
    loading.classList.remove('active');
    searchBtn.disabled = false;
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.add('active');
    setTimeout(() => hideError(), 5000);
}

function hideError() {
    errorMessage.classList.remove('active');
}

function hideResults() {
    resultsSection.classList.remove('active');
    emptyState.style.display = 'flex';
    videoPreview.classList.remove('active');
    if (player && typeof player.stopVideo === 'function') {
        player.stopVideo();
    }
}

// Initialize
console.log('Image Retrieval UI initialized');
console.log('API Base URL:', API_BASE_URL);