/* Description: Custom JS file */

/* Navigation*/
// Collapse the navbar by adding the top-nav-collapse class
window.onscroll = function () {
	scrollFunction();
};

function scrollFunction() {
	let intViewportWidth = window.innerWidth;
	if (
		document.body.scrollTop > 30 ||
		(document.documentElement.scrollTop > 30) & (intViewportWidth > 991)
	) {
		document.getElementById("navbar").classList.add("top-nav-collapse");
	} else if (
		document.body.scrollTop < 30 ||
		(document.documentElement.scrollTop < 30) & (intViewportWidth > 991)
	) {
		document.getElementById("navbar").classList.remove("top-nav-collapse");
	}
}

// Navbar on mobile
let elements = document.querySelectorAll(".nav-link:not(.dropdown-toggle)");

for (let i = 0; i < elements.length; i++) {
	elements[i].addEventListener("click", () => {
		document.querySelector(".offcanvas-collapse").classList.toggle("open");
	});
}

document.querySelector(".navbar-toggler").addEventListener("click", () => {
  	document.querySelector(".offcanvas-collapse").classList.toggle("open");
});

// /* Card Slider - Swiper */
// var cardSlider = new Swiper('.card-slider', {
// 	autoplay: {
// 		delay: 4000,
// 		disableOnInteraction: false
// 	},
// 	loop: true,
// 	navigation: {
// 		nextEl: '.swiper-button-next',
// 		prevEl: '.swiper-button-prev'
// 	}
// });


/* Modal Button Close And Scroll To Link */
const theModal = document.getElementById('staticBackdrop');
// Needs the if otherwise it will serve error on pages without the modal like the extra pages
if (theModal !== null) { 
	const theModalGen = new bootstrap.Modal(theModal);
	const theModalCtaBtn = document.getElementById('modalCtaBtn');
	theModalCtaBtn.addEventListener('click', function () {
		theModalGen.hide(); 
	})
}


// When the user clicks on the button, scroll to the top of the document
function topFunction() {
	document.body.scrollTop = 0; // for Safari
	document.documentElement.scrollTop = 0; // for Chrome, Firefox, IE and Opera
}