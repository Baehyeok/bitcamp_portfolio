
/*
    Mode Option template
*/
Vue.component('df-option-mode',{
    template:`
    <section class="pricing-card-area section-bg" data-background="assets/img/gallery/section_bg1.png">
        <div class="container">
                <!-- Section Tittle -->
                <div class="row d-flex justify-content-center">
                <div class="col-lg-8">
                    <div class="section-tittle text-center mb-70">
                        <h2 style = "color : #A65BF8 ">Play Mode</h2>
                    </div>
                </div>
            </div>
            <section class="class1">
                <div class="row align-items-end">
                    <div class="col-xl-4 col-lg-4 col-md-6" style="width: 330px !important;">
                        <div id="practice-card" class="mode-option-card single-card text-center mb-30" data-value="practice" style="
                        padding-left: 0px;
                        padding-right: 0px;
                        border-left-width: 0px;
                        border-right-width: 0px;
                        padding-top: 30px;
                    ">
                            <div class="card-top">
                                <span></span>
                                <h4>Practice</h4>
                            </div>
                            <div class="card-bottom" >
                                <ul>
                                    <div style="background-color: #C1A0D9;height: 143px;padding-top: 5px;">
                                    <br/><br/>
                                    <strong><h6>Real Time Mode or Upload Your Video</h6></string>
                                </div>
                                </br>
                                </ul>
                            </div>
                            <div class="card-buttons mt-30 select-btn-line">
                                <a  class="btn card-btn1 select-btn" >Select</a>
                            </div>
                        </div>
                    </div>
                   
                    <div class="col-xl-4 col-lg-4 col-md-6" style="width: 330px !important;">
                        <div id="random-card" class="mode-option-card single-card text-center mb-30 disable-option" data-value="random" style="
                        padding-left: 0px;
                        padding-right: 0px;
                        border-left-width: 0px;
                        border-right-width: 0px;
                        padding-top: 30px;
                    ">
                            <div class="card-top">
                                <span></span>
                                <h4>Random</h4>
                            </div>
                            <div class="card-bottom">
                                <ul>
                                    <div style="background-color: #C1A0D9; height: 144px;padding-top: 8px;" class="disable-option">
                                        <strong>　</strong></br></br>
                                        <strong><h6>Only Real Time Mode</h6></strong></br>
                                        <strong>　</strong>
                                    </div>
                                </br>
                                </ul>
                            </div>
                            <div class="card-buttons mt-30 select-btn-line disable-option">
                                <a  class="btn card-btn1  disable-option"></a>
                            </div> 
                        </div>
                    </div>

                </section>
            </div>
        </div>
    </section>
    `
});



let app_option_mode = new Vue({
    el: "#app-option-mode",
    delimiters : ['[[', ']]'],
});


/*
    Upload Option Template
*/
Vue.component('df-option-upload', {
    template:`
    <section class="pricing-card-area section-bg" data-background="assets/img/gallery/section_bg1.png">
        <div class="container">
                <!-- Section Tittle -->
                <div class="row d-flex justify-content-center">
                <div class="col-lg-8">
                    <div class="section-tittle text-center mb-70">
                        <h2 style = "color : #A65BF8 ">Play Mode</h2>
                    </div>
                </div>
            </div>
            <div class="row align-items-end">
                <div class="col-xl-4 col-lg-4 col-md-6" style="width: 330px !important;">
                    <div id="realtime-card" class="upload-option-card single-card text-center mb-30" data-value="realtime" style="padding-top: 30px;padding-right: 0px;padding-left: 0px;border-right-width: 0px;border-left-width: 0px;">
                        <div class="card-top">
                            <span></span>
                            <h4>Real time</h4>
                        </div>
                        <div class="card-bottom">
                            <ul>
                                <div style="background-color: #C1A0D9;height: 150px;">
                                    <br/><br/>
                                    <strong><h6>Setting Your WebCam</h6></strong></br></br>
                                </div>
                            </ul>
                        </div>
                        <div class="card-buttons mt-30 select-btn-line">
                            <a  class="btn card-btn1 select-btn">Select</a>
                        </div>
                    </div>
                </div>
            
                <div class="col-xl-4 col-lg-4 col-md-6" style="width: 330px !important;">
                    <div id="upload-card" class="upload-option-card single-card text-center mb-30" data-value="upload" style="padding-top: 30px;padding-right: 0px;padding-left: 0px;border-right-width: 0px;border-left-width: 0px;border-bottom-width: 26px;">
                        <div class="card-top">
                            <span></span>
                            <h4>Upload</h4>
                        </div>
                        <div class="card-bottom">
                            <ul>
                                <li id='droped-file-zone'>　</li>
                                <div class="file-drop-zone" style="height: 150px;">
                                    <br/><br/>
                                    <strong><h6>Drag & Drop Your Video</h6></strong></br></br>
                                    <!-- <li><button style = "background-color: #211A4D ">Choose a File</button></li> -->
                                </div>
                            </ul>
                        </div>
                        <div class="card-buttons mt-30 select-btn-line" >
                            <a  class="btn card-btn1 select-btn">Select</a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>
    `
});

let app_option_upload = new Vue({
    el: '#app-option-upload',
    delimiters : ['[[', ']]'],
})


/*
    Display Option Template
*/

Vue.component('df-option-display', {
    template:`
    <section class="pricing-card-area section-bg" data-background="assets/img/gallery/section_bg1.png">
        <div class="container">
                <!-- Section Tittle -->
                <div class="row d-flex justify-content-center">
                <div class="col-lg-8">
                    <div class="section-tittle text-center mb-70">
                        <h2 style = "color : #A65BF8 ">Display Mode</h2>
                        
                        <!-- <p style = "color : #1994AF">Select Your Display Mode</p> -->
                        
                    </div>
                </div>
            </div>
            <div class="row align-items-end">
                <div class="col-xl-4 col-lg-4 col-md-6" >
                    <div id="u-p-h-card" class="display-option-card single-card text-center mb-30" data-value="uph" style="
                    padding-top: 30px;
                ">
                        <div class="card-top">
                            <span></span>
                            <h4>Vertical Half</h4>
                        </div>
                        <div class="card-bottom">
                            <ul>
                                <div style="display: flex;justify-content: center;">
                                <button style="background-color: #211A4D;width: 161px;height: 103px;">1</button>
                                <button style="background-color: white; width: 161px;height: 103px; color: black;">2</button>
                                 </div>
                            </br>

                            <li>　</li>
                            </ul>
                        </div>
                        <div class="card-buttons mt-30 select-btn-line">
                            <a  class="btn card-btn1 select-btn"  >Select</a>
                        </div>
                    </div>
                </div>
                <div class="col-xl-4 col-lg-4 col-md-6">
                    <div id="u-p-v-card" class="display-option-card single-card text-center mb-30  disable-option" data-value="upv" style="
                    padding-top: 30px;
                ">
                        <div class="card-top">
                            <span></span>
                            <h4>Horizontal Half</h4>
                        </div>
                        <div class="card-bottom">
                            <ul>
                                <div style="display: flex;flex-direction: column;">
                                <button style="background-color: #211A4D;height: 52px;">1</button>
                                <button style="background-color: white;height: 52px; color: black;">2</button>
                            </div>
                            </br>

                            <li>　</li>
                            </ul>
                        </div>
                        <div class="card-buttons mt-30 select-btn-line disable-option">
                            <a  class="btn card-btn1 disable-option"  ></a>
                        </div>
                    </div>
                </div>
                <div class="col-xl-4 col-lg-4 col-md-6">
                    <div id="u-o-card" class="display-option-card single-card text-center mb-30 disable-option" data-value="uo" style="
                    padding-top: 30px;
                ">
                        <div class="card-top">
                            <span></span>
                            <h4>Model Only</h4>
                        </div>
                        <div class="card-bottom">
                            <ul>
                                <div style="display: flex;flex-direction: column;">
                                <button style="background-color: #211A4D;height: 106px;">1</button>
                            </div>
                            </br>

                            <li>　</li>
                            </ul>
                        </div>
                        <div class="card-buttons mt-30 select-btn-line disable-option">
                            <a  class="btn card-btn1 disable-option"  ></a>
                        </div>
                       
                    </div>
                </div>
    </section>
     
    `
});


let app_option_display = new Vue({
    el : '#app-option-display',
    delimiters : ['[[', ']]'],
})

Vue.component('df-option-song',{
    template:`
    
    <!-- Choose your song -->
    <div class="wrap">
        <div class="wrapper">
            <h2 class="title">Choose your song</h2>
            <div class="wrapper_a">
                <div class="wrap_checkbox">
                    <input type="checkbox" class="toggle" id="rounded">
                    <label for="rounded" data-checked="K-POP" class="rounded" data-unchecked="POP"></label>
                </div>
                <div class="search">
                    <input type="text"/>
                    <div class="fas fa-search"></div>
                </div>
            </div>
        </div>
        <!-- K-POP items -->
        <div class="k-pop_items">
            <div class="container_1">
                <label class="option_item">
                    <input type="checkbox" class="checkbox">
                    <div class="option_inner Akmu ">
                        <div class="tickmark"></div>
                        <div class="icon"></div>
                        <i class="fal fa-badge-dollar"></i>
                        <div class="name">
                            <div class="song-title">Falling</div>
                            <div class="artist">Akmu</div>
                        </div>
                    </div>
                </label>
                <label class="option_item">
                    <input type="checkbox" class="checkbox">
                    <div class="option_inner BTS">
                        <div class="tickmark"></div>
                        <div class="icon"></div>
                        <i class="fal fa-badge-dollar"></i>
                        <div class="name">
                            <div class="song-title">Permission to Dance</div>
                            <div class="artist">BTS</div>
                        </div>
                    </div>
                </label>
                <label class="option_item">
                    <input type="checkbox" class="checkbox">
                    <div class="option_inner Redvelvet">
                        <div class="tickmark"></div>
                        <div class="icon"></div>
                        <i class="fal fa-badge-dollar"></i>
                        <div class="name">
                            <div class="song-title">Queendom</div>
                            <div class="artist">Red Velvet</div>
                        </div>
                    </div>
                </label>
                <label class="option_item">
                    <input type="checkbox" class="checkbox">
                    <div class="option_inner BTS">
                        <div class="tickmark"></div>
                        <div class="icon"></div>
                        <i class="fal fa-badge-dollar"></i>
                        <div class="name">
                            <div class="song-title">Butter</div>
                            <div class="artist">BTS</div>
                        </div>
                    </div>
                </label>
                <label class="option_item">
                    <input type="checkbox" class="checkbox">
                    <div class="option_inner IU">
                        <div class="tickmark"></div>
                        <div class="icon"></div>
                        <i class="fal fa-badge-dollar"></i>
                        <div class="name">
                            <div class="song-title">Lilak</div>
                            <div class="artist">IU</div>
                        </div>
                    </div>
                </label>
                <label class="option_item">
                    <input type="checkbox" class="checkbox">
                    <div class="option_inner Bravegirls">
                        <div class="tickmark"></div>
                        <div class="icon"></div>
                        <i class="fal fa-badge-dollar"></i>
                        <div class="name">
                            <div class="song-title">Chi Mat Ba Ram</div>
                            <div class="artist">Brave Girls</div>
                        </div>
                    </div>
                </label>
                <label class="option_item">
                    <input type="checkbox" class="checkbox">
                    <div class="option_inner Joy">
                        <div class="tickmark"></div>
                        <div class="icon"></div>
                        <i class="fal fa-badge-dollar"></i>
                        <div class="name">
                            <div class="song-title">Hello</div>
                            <div class="artist">Joy</div>
                        </div>
                    </div>
                </label>
                <label class="option_item">
                    <input type="checkbox" class="checkbox">
                    <div class="option_inner Homider">
                        <div class="tickmark"></div>
                        <div class="icon"></div>
                        <i class="fal fa-badge-dollar"></i>
                        <div class="name">
                            <div class="song-title">Siren</div>
                            <div class="artist">HOMIDER</div>
                        </div>
                    </div>
                </label>
            </div>
        </div>

        <!-- POP items -->
        <div class="POP_items">
            <div class="container_1">
                <label class="option_item">
                    <input type="checkbox" class="checkbox">
                    <div class="option_inner Justinbieber">
                        <div class="tickmark"></div>
                        <div class="icon"></div>
                        <i class="fal fa-badge-dollar"></i>
                        <div class="name">
                            <div class="song-title">Stay</div>
                            <div class="artist">Justin Bieber</div>
                        </div>
                    </div>
                </label>
                <label class="option_item">
                    <input type="checkbox" class="checkbox">
                    <div class="option_inner Jawsh685">
                        <div class="tickmark"></div>
                        <div class="icon"></div>
                        <i class="fal fa-badge-dollar"></i>
                        <div class="name">
                            <div class="song-title">Savage Love</div>
                            <div class="artist">Jawsh 685</div>
                        </div>
                    </div>
                </label>
                <label class="option_item">
                    <input type="checkbox" class="checkbox">
                    <div class="option_inner Annemarie">
                        <div class="tickmark"></div>
                        <div class="icon"></div>
                        <i class="fal fa-badge-dollar"></i>
                        <div class="name">
                            <div class="song-title">2002</div>
                            <div class="artist">Anne-Marie</div>
                        </div>
                    </div>
                </label>
                <label class="option_item">
                    <input type="checkbox" class="checkbox">
                    <div class="option_inner Maroon5">
                        <div class="tickmark"></div>
                        <div class="icon"></div>
                        <i class="fal fa-badge-dollar"></i>
                        <div class="name">
                            <div class="song-title">Memories</div>
                            <div class="artist">Marron5</div>
                        </div>
                    </div>
                </label>
                <label class="option_item">
                    <input type="checkbox" class="checkbox">
                    <div class="option_inner Dojacat">
                        <div class="tickmark"></div>
                        <div class="icon"></div>
                        <i class="fal fa-badge-dollar"></i>
                        <div class="name">
                            <div class="song-title">Need to Know</div>
                            <div class="artist">Doja Cat
                            </div>
                        </div>
                    </div>
                </label>
                <label class="option_item">
                    <input type="checkbox" class="checkbox">
                    <div class="option_inner Oliviarodrigo">
                        <div class="tickmark"></div>
                        <div class="icon"></div>
                        <i class="fal fa-badge-dollar"></i>
                        <div class="name">
                            <div class="song-title">Good 4 u</div>
                            <div class="artist">Olivia Rodrigo</div>
                        </div>
                    </div>
                </label>
                <label class="option_item">
                    <input type="checkbox" class="checkbox">
                    <div class="option_inner Billieeilish">
                        <div class="tickmark"></div>
                        <div class="icon"></div>
                        <i class="fal fa-badge-dollar"></i>
                        <div class="name">
                            <div class="song-title">Happier Than ever</div>
                            <div class="artist">Billie Eilish</div>
                        </div>
                    </div>
                </label>
                <label class="option_item">
                    <input type="checkbox" class="checkbox">
                    <div class="option_inner Lilnas">
                        <div class="tickmark"></div>
                        <div class="icon"></div>
                        <i class="fal fa-badge-dollar"></i>
                        <div class="name">
                            <div class="song-title">Industry Baby</div>
                            <div class="artist">Lil Nas</div>
                        </div>
                    </div>
                </label>
            </div>
        </div>

    </div>    
    `
})



let app_option_song = new Vue({
    el : '#app-option-song',
    delimiters : ['[[', ']]'],
})
