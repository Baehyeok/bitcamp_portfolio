


/*
    Index page navigation template
*/
Vue.component('df-index-nav', {
    template : ` <!-- Navigation -->
    <nav id="navbar" class="navbar navbar-expand-lg fixed-top navbar-dark" aria-label="Main navigation">
        <div class="container">
            <!-- Image Logo -->
            <a class="navbar-brand logo-image" href="/"><img src="/static/images/logo.png" alt="alternative"><span></span></a> 

            <button class="navbar-toggler p-0 border-0" type="button" id="navbarSideCollapse" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>

            <div class="navbar-collapse offcanvas-collapse" id="navbarsExampleDefault">
                <ul class="navbar-nav ms-auto navbar-nav-scroll">
                    <li class="nav-item">
                        <a class="nav-link active" aria-current="page" href="#header">Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#community">Community</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" aria-current="page" href="#aboutus">About us</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#contact">Contact</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#howto">Ranking</a>
                    </li>
                    <li class="nav-item">
                        {% if request.session.user %}
                        <!-- 유저 로고 만들고 유저페이지 이동 및 로그아웃 만들기 -->
                        <a class="btn-solid-reg sign-btn" href="/user/logout">logout</a>
                        {% else %}
                        <a class="btn-solid-reg sign-btn" href="/user/login">login</a>
                        {% endif %}
                    </li>
                </ul>
            </div> <!-- end of navbar-collapse -->
        </div> <!-- end of container -->
    </nav> <!-- end of navbar -->
    <!-- end of navigation -->`,

});



let app_index_nav = new Vue({
    el:'#app-index-nav',
    delimiters : ['[[', ']]'],
});



/*
    Main-Content-Service page navigation template
*/
Vue.component('df-main-nav', {
    template : ` <!-- Navigation -->
    <nav id="navbar" class="navbar navbar-expand-lg fixed-top navbar-dark" aria-label="Main navigation">
        <div class="container">
            <!-- Image Logo -->
            <a class="navbar-brand logo-image" href="/"><img src="/static/images/logo.png" alt="alternative"><span></span></a> 

            <button class="navbar-toggler p-0 border-0" type="button" id="navbarSideCollapse" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>

            <div class="navbar-collapse offcanvas-collapse" id="navbarsExampleDefault">
                <ul class="navbar-nav ms-auto navbar-nav-scroll">
                    <li class="nav-item">
                        <a class="nav-link active" aria-current="page" href="/">Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/community">Community</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" aria-current="page" href="/play/option">Play</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#Ranking">Ranking</a>
                    </li>
                    <li class="nav-item">
                        <a class="btn-solid-reg sign-btn" href="/sign" >User</a>
                    </li>
                </ul>
            </div> <!-- end of navbar-collapse -->
        </div> <!-- end of container -->
    </nav> <!-- end of navbar -->
    <!-- end of navigation -->`,

});



let app_main_nav = new Vue({
    el:'#app-main-nav',
    delimiters : ['[[', ']]'],
});



/*
    Play page navigation template
*/
Vue.component('df-play-nav', {
    template : `<!-- Navigation -->
    <nav id="navbar" class="hide navbar navbar-expand-lg fixed-top navbar-dark" aria-label="Main navigation">
        <div class="container">
            <!-- Image Logo -->
            <a class="navbar-brand logo-image " href="/"><img class="" src="/static/images/logo.png" alt="alternative"><span></span></a> 

            <button class="navbar-toggler p-0 border-0" type="button" id="navbarSideCollapse" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>

            <div class="navbar-collapse offcanvas-collapse" id="navbarsExampleDefault">
                <ul class="navbar-nav ms-auto navbar-nav-scroll">
                    <li class="nav-item">
                        <!-- <a class="btn-solid-reg sign-btn" href="/sign" >Sign in</a> -->
                    </li>
                </ul>
            </div> <!-- end of navbar-collapse -->
        </div> <!-- end of container -->
    </nav> <!-- end of navbar -->
    <!-- end of navigation -->
  `,

});



let app_play_nav = new Vue({
    el:'#app-play-nav',
    delimiters : ['[[', ']]'],
});







