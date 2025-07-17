// Mobile Navigation Toggle
document.addEventListener('DOMContentLoaded', function() {
    const navToggle = document.getElementById('nav-toggle');
    const navMenu = document.getElementById('nav-menu');
    const lightbox = document.getElementById('lightbox');
    const lightboxClose = document.getElementById('lightbox-close');
    const searchBtn = document.getElementById('search-btn');
    const searchInput = document.getElementById('search-input');

    // Mobile menu toggle
    navToggle.addEventListener('click', function() {
        navMenu.classList.toggle('active');
    });

    // Close mobile menu when clicking on a link
    document.querySelectorAll('.nav-menu a').forEach(link => {
        link.addEventListener('click', () => {
            navMenu.classList.remove('active');
        });
    });

    // Load cartoons
    loadCartoons();

    // Search functionality
    searchBtn.addEventListener('click', performSearch);
    searchInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            performSearch();
        }
    });

    // Lightbox functionality
    lightboxClose.addEventListener('click', closeLightbox);
    lightbox.addEventListener('click', function(e) {
        if (e.target === lightbox) {
            closeLightbox();
        }
    });

    // ESC key to close lightbox
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeLightbox();
        }
    });
});

// Sample cartoon data (in real implementation, this would come from a backend)
const cartoons = [
    {
        id: 1,
        title: "Political Commentary #1",
        description: "A thought-provoking illustration about current events",
        image: "assets/cartoon1.jpg",
        tags: ["politics", "commentary", "current-events"]
    },
    {
        id: 2,
        title: "Economic Perspective",
        description: "Visual commentary on economic policies and their impact",
        image: "assets/cartoon2.jpg",
        tags: ["economics", "policy", "finance"]
    },
    {
        id: 3,
        title: "Social Issues Focus",
        description: "Artistic interpretation of contemporary social challenges",
        image: "assets/cartoon3.jpg",
        tags: ["social", "society", "culture"]
    },
    {
        id: 4,
        title: "Government Critique",
        description: "Critical analysis through satirical illustration",
        image: "assets/cartoon4.jpg",
        tags: ["government", "satire", "critique"]
    },
    {
        id: 5,
        title: "Media Analysis",
        description: "Commentary on media representation and bias",
        image: "assets/cartoon5.jpg",
        tags: ["media", "journalism", "bias"]
    },
    {
        id: 6,
        title: "International Relations",
        description: "Global perspective on diplomatic affairs",
        image: "assets/cartoon6.jpg",
        tags: ["international", "diplomacy", "global"]
    }
];

let currentCartoons = [...cartoons];

function loadCartoons(cartoonsToShow = currentCartoons) {
    const cartoonGrid = document.getElementById('cartoon-grid');
    cartoonGrid.innerHTML = '';

    cartoonsToShow.forEach(cartoon => {
        const cartoonElement = createCartoonElement(cartoon);
        cartoonGrid.appendChild(cartoonElement);
    });
}

function createCartoonElement(cartoon) {
    const cartoonDiv = document.createElement('div');
    cartoonDiv.className = 'cartoon-item';
    cartoonDiv.innerHTML = `
        <img src="${cartoon.image}" alt="${cartoon.title}" onerror="this.src='assets/placeholder.jpg'">
        <div class="cartoon-info">
            <h4>${cartoon.title}</h4>
            <p>${cartoon.description}</p>
        </div>
    `;

    cartoonDiv.addEventListener('click', () => openLightbox(cartoon));

    return cartoonDiv;
}

function openLightbox(cartoon) {
    const lightbox = document.getElementById('lightbox');
    const lightboxImage = document.getElementById('lightbox-image');
    const lightboxTitle = document.getElementById('lightbox-title');
    const lightboxDescription = document.getElementById('lightbox-description');

    lightboxImage.src = cartoon.image;
    lightboxImage.alt = cartoon.title;
    lightboxTitle.textContent = cartoon.title;
    lightboxDescription.textContent = cartoon.description;

    lightbox.style.display = 'flex';
    document.body.style.overflow = 'hidden';
}

function closeLightbox() {
    const lightbox = document.getElementById('lightbox');
    lightbox.style.display = 'none';
    document.body.style.overflow = 'auto';
}

function performSearch() {
    const searchTerm = document.getElementById('search-input').value.toLowerCase().trim();

    if (searchTerm === '') {
        currentCartoons = [...cartoons];
    } else {
        currentCartoons = cartoons.filter(cartoon =>
            cartoon.title.toLowerCase().includes(searchTerm) ||
            cartoon.description.toLowerCase().includes(searchTerm) ||
            cartoon.tags.some(tag => tag.toLowerCase().includes(searchTerm))
        );
    }

    loadCartoons(currentCartoons);

    // Show search results message
    const cartoonGrid = document.getElementById('cartoon-grid');
    if (currentCartoons.length === 0) {
        cartoonGrid.innerHTML = '<p style="text-align: center; grid-column: 1 / -1; padding: 2rem; color: #666;">No cartoons found matching your search.</p>';
    }
}

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add loading animation
function showLoading() {
    const cartoonGrid = document.getElementById('cartoon-grid');
    cartoonGrid.innerHTML = '<div style="text-align: center; grid-column: 1 / -1; padding: 2rem;"><i class="fas fa-spinner fa-spin fa-2x"></i></div>';
}

// Intersection Observer for lazy loading (future enhancement)
const observerOptions = {
    root: null,
    rootMargin: '50px',
    threshold: 0.1
};

const imageObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const img = entry.target;
            img.src = img.dataset.src;
            img.classList.remove('lazy');
            observer.unobserve(img);
        }
    });
}, observerOptions);

// Initialize lazy loading for images
function initLazyLoading() {
    const lazyImages = document.querySelectorAll('img[data-src]');
    lazyImages.forEach(img => imageObserver.observe(img));
}