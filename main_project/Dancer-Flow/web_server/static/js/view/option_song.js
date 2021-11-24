const searchEl= document.querySelector('.search');
const searchinputEl = searchEl.querySelector('input');

searchEl.addEventListener('click', function () {
  searchinputEl.focus();
});

searchinputEl.addEventListener('focus', function () {
  searchEl.classList.add('focused');
  searchinputEl.setAttribute('placeholder', 'Search');
});

searchinputEl.addEventListener('focus', function () {
  searchEl.classList.remove('focused');
  searchinputEl.setAttribute('placeholder', '');
});

