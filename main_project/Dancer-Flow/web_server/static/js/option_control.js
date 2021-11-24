/*
    Option Card Elements 
*/
const upload_card = document.querySelector('#upload-card');
const cards = document.querySelectorAll('.single-card')     // card list
const song_cards = document.querySelectorAll('.option_item') // songs
const option_container = document.querySelectorAll('.option-container')     // option view

/*
    Button
 */ 
const prev_btn = document.querySelector('#prev-option-btn');
const next_btn = document.querySelector('#next-option-btn');
const start_btn = document.querySelector('#play-start-btn');
const select_btn = document.querySelectorAll('.select-btn')


/*
    Setting Values
*/
const ITEM_FOCUSED_CLASSNAME = 'item-focused';
const SONG_SELECTED_CLASSNAME = 'song-selected';

const OPTION_SELECTOR = {
    'MODE': '.mode-option-card',
    'UPLOAD': '.upload-option-card',
    'DISPLAY': '.display-option-card',
    'SONG' : '.display-option-song',
    'MODE_C' : '#app-option-mode',
    'UPLOAD_C' : '#app-option-upload',
    'DISPLAY_C' : '#app-option-display',
    'SONG_C' : '#app-option-song',
}
let visible_target_option  = OPTION_SELECTOR.MODE; 
let visible_target  = OPTION_SELECTOR.MODE_C; 

/*
    Option Form Value
*/
let options = {
    'mode' : '',
    'upload' : '',
    'upload_file' : '',
    'display' : '',
    'songs' : [],
}

/*
    Card Event
*/
function switch_card(item){    
    const sets = document.querySelectorAll(visible_target_option)
    sets.forEach(i=>i.classList.remove(ITEM_FOCUSED_CLASSNAME));
    item.classList.add(ITEM_FOCUSED_CLASSNAME);
}

cards.forEach(item =>
    item.addEventListener('click', e => {
        if (e.currentTarget.classList.contains('disable-option')){
            return;
        } 
        
        switch_card(item);
        if (Array.from(select_btn).includes(e.target)){
            next_btn.click()
        }
}));
cards.forEach(item => 
    item.addEventListener('dblclick', e => {
        if (e.currentTarget.classList.contains('disable-option')){
            return;
        } 
        switch_card(item);
        next_btn.click();
}));

// browser 기본 drag drop 이벤트(ex file open) 를 막기 위해 preventdefault 호출
upload_card.addEventListener('dragover', e=>{
    e.preventDefault()
    upload_card.style.backgroundColor = '#C1A0D9'  
})
upload_card.addEventListener('dragleave', e=>{
    e.preventDefault()
    upload_card.style.backgroundColor = '#ffffff'  
})
upload_card.addEventListener('drop', e=>{
    e.preventDefault()
    console.log(e.dataTransfer.files)
    const video_file = e.dataTransfer.files[0]
    options.upload_file = video_file
    
    const data_zone = document.getElementById('droped-file-zone')
    data_zone.innerHTML = `<h6> ${video_file.name} Droped </h6>`
    e.target.click()
    next_btn.click()
})

song_cards.forEach(item=>item.addEventListener('click', e=>{
    const SONG_ID = e.currentTarget.getElementsByClassName('song_item')[0].getAttribute('data-id')   
    if(e.currentTarget.getElementsByClassName('checkbox')[0].checked){
        e.currentTarget.classList.add(SONG_SELECTED_CLASSNAME)
        options.songs.push(SONG_ID)
    }else{
        e.currentTarget.classList.remove(SONG_SELECTED_CLASSNAME)
        options.songs = options.songs.filter(song=>song !== SONG_ID)
    }
    console.log(options.songs)
}))

/*
    Button Event
*/
function switch_option (target_option) {
    option_container.forEach(i=>i.classList.remove('content-visible'))
    option_container.forEach(i=>i.classList.add('content-hide'))
    target_option.classList.add('content-visible')
}

select_btn.forEach(btn=>btn.addEventListener('click', e=>{
    // console.log('selected')
}))
prev_btn.addEventListener('click',e=>{
    switch(visible_target_option){
        case OPTION_SELECTOR.MODE:  
            console.error('logic error')
            break;
        case OPTION_SELECTOR.UPLOAD:
            switch_option(option_container[0])
            prev_btn.classList.remove('content-visible')
            prev_btn.classList.add('content-hide')
            visible_target_option = OPTION_SELECTOR.MODE;
            visible_target = OPTION_SELECTOR.MODE_C
            break;
        case OPTION_SELECTOR.DISPLAY:
            switch_option(option_container[1])
            visible_target_option = OPTION_SELECTOR.UPLOAD;
            visible_target = OPTION_SELECTOR.UPLOAD_C
            break;
        case OPTION_SELECTOR.SONG:
            switch_option(option_container[2])
            next_btn.classList.add('content-visible')
            next_btn.classList.remove('content-hide')
            start_btn.classList.remove('content-visible')
            start_btn.classList.add('content-hide')
            visible_target_option = OPTION_SELECTOR.DISPLAY;
            visible_target = OPTION_SELECTOR.DISPLAY_C
            break;
    }
});

next_btn.addEventListener('click',e=>{
    const target = document.querySelector(visible_target)
    if (target.getElementsByClassName(ITEM_FOCUSED_CLASSNAME).length !== 1){
        window.alert('옵션을 선택해주세요')
        return
    }
    if(upload_card.classList.contains(ITEM_FOCUSED_CLASSNAME) && options.upload_file === '' && visible_target === OPTION_SELECTOR.UPLOAD_C){
        window.alert('비디오를 업로드 해주세요.')
        return
    }
    switch(visible_target_option){
        case OPTION_SELECTOR.MODE:  
            switch_option(option_container[1])  
            prev_btn.classList.add('content-visible')
            visible_target_option = OPTION_SELECTOR.UPLOAD;
            visible_target = OPTION_SELECTOR.UPLOAD_C
            break;
        case OPTION_SELECTOR.UPLOAD:
            switch_option(option_container[2])
            visible_target_option = OPTION_SELECTOR.DISPLAY;
            visible_target = OPTION_SELECTOR.DISPLAY_C
            break;
        case OPTION_SELECTOR.DISPLAY:
            switch_option(option_container[3])
            next_btn.classList.remove('content-visible')
            next_btn.classList.add('content-hide')
            start_btn.classList.add('content-visible')
            start_btn.classList.remove('content-hide')
            visible_target_option = OPTION_SELECTOR.SONG;
            visible_target = OPTION_SELECTOR.SONG_C
            
            break;
        case OPTION_SELECTOR.SONG:
            console.error('logic error') 
            break;
    }
});

// Submit
start_btn.addEventListener('click', async e=>{
    const option_list = document.getElementsByClassName(ITEM_FOCUSED_CLASSNAME)
    let count = option_list.length + options.songs.length;

    if(count < 4){
        window.alert('옵션을 모두 설정해주세요.');
        return
    }
    if(options.songs === [] || options.length < 1){
        window.alert('곡을 선택해주세요.')
        return
    }
    const data = new FormData()
    if(upload_card.classList.contains(ITEM_FOCUSED_CLASSNAME)){
        data.append('video', options.upload_file, options.upload_file.name)
    }else{
        data.append('video', "")
    }
    data.append('mode', option_list[0].getAttribute('data-value'))
    data.append('upload', option_list[1].getAttribute('data-value'))
    data.append('display', option_list[2].getAttribute('data-value'))
    data.append('songs', JSON.stringify(options.songs))
    console.log(data)
    let res = await sendServer('http://127.0.0.1:8000/play/option/', data)
    console.log(res)
    if(res.redirected){
        window.location.assign(res.url)
    }
})


/*
    Server Request
*/
async function sendServer(URL, datas){
    const request = new Request(URL + '', {
        headers:{
            'X-CSRFToken': getCookie("csrftoken"),
        }
    })
    let res = await fetch(request, {
        method : 'POST',
        mode : 'same-origin',
        body : datas,
    })
    return res
}

function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}