const view_cards = document.querySelectorAll('.view-card')
const coins = document.querySelectorAll('.coins')


view_cards.forEach(item=>{ 
    item.addEventListener('click', async e=>{
        const PID = e.currentTarget.getAttribute('data-pid')             
        let type = e.target.getAttribute('data-type')
        let faves = e.currentTarget.getElementsByClassName('faves-text')[0]
        if(Array.from(coins).includes(e.target)){
            const data = new FormData()
            data.append('pid', PID)
            data.append('type', type)
            let res = await sendServer('http://127.0.0.1:8000/community/', data)
            // let res_data = await res.json()
            res.json().then(data=>{
                if (data.result === 200){
                    if (type === 'on'){
                        e.target.setAttribute('src', '/static/images/coin-off.png')
                        e.target.setAttribute('data-type', 'off')
                        console.log(e.currentTarget)
                        faves.innerText = parseInt(faves.textContent) - 1 >= 0 ? parseInt(faves.textContent) - 1 : 0
                    }else{
                        e.target.setAttribute('src', '/static/images/coin-on.png')
                        e.target.setAttribute('data-type', 'on')
                        faves.innerText = parseInt(faves.textContent) + 1
                    }
                    if(data.result == 200){
                        console.log('asdf')
                    }
                }
            })
        }else{
            window.location.assign(`/community/view/${PID}`)
        }
        
    })
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