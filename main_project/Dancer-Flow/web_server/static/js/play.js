

class PlayView{
    constructor(){
        this.color_frame = document.getElementById('canvas-container')
        this.canvas = document.getElementById('userCanvas'),
        this.context = this.canvas.getContext('2d'),
        this.playCanvas = document.getElementById('playCanvas')
        this.skeletonCanvas = document.getElementById('skeletonCanvas')
        this.skeletonContext = this.skeletonCanvas.getContext('2d')
        this.score = 0
        this.recordCallback = null
        this.canvas.width = window.screen.width * 0.95
        this.canvas.height = window.screen.height*0.55
        this.skeletonCanvas.width = window.screen.width * 0.21
        this.skeletonCanvas.height = window.screen.height * 0.18
        this.diff = 0
        this.count = 5
        this.skeleton_image = ''
        this.loopID = null
    }
    setRecordCallback(callback){
        this.recordCallback = callback
    }
    resizeCanvas() {
        console.log('resizing')
        let width = parseInt(window.innerWidth * 1.);
        let height = parseInt(window.innerHeight * 0.45);

        this.canvas.width = width;
        this.canvas.height = height;
    }
    
    async draw_intro(count=3, mode, user_video, play_video){

        this.context.strokeStyle= '#a65bf8'
        this.context.lineWidth = 6
        this.context.fillStyle = 'white'

        async function intro(count){
            this.context.clearRect(0,0, this.canvas.width, this.canvas.height);

            this.context.font = `${0.004*this.canvas.width}rem  Roboto Slab, serif`;
            this.context.strokeText(`Ready...`, (this.canvas.width * 0.818) /2, (this.canvas.height*0.798) / 2);
            this.context.fillText(`Ready...`, (this.canvas.width * 0.818) /2, (this.canvas.height * 0.798) /2);
    
            if (count > 0){
                this.context.strokeText(`${count}`, (this.canvas.width * 0.919) /2, (this.canvas.height * 1.2519) /2);
                this.context.fillText(`${count}`, (this.canvas.width * 0.919) /2, (this.canvas.height * 1.2519) /2);
            }
            else{
                this.context.strokeText(`Start !`, (this.canvas.width * 0.819) /2, (this.canvas.height * 1.2519) /2);
                this.context.fillText(`Start !`, (this.canvas.width * 0.819) /2, (this.canvas.height * 1.2519) /2);
            }             

            if (count < 0) {
                this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
                play_video.play()
                user_video.play()
                this.loopID = window.requestAnimationFrame(timestamp => {
                    this.draw_play.bind(this, timestamp, mode, user_video, play_video, this.score)()
                    this.recordCallback()
                })
            }
            else{
                setTimeout(intro.bind(this), 1000, count-1)
            }
        }
        
        setTimeout(intro.bind(this), 1000, count)
    }
    draw_play(timestamp, mode, user_video, play_video, total_score) {        

        
        // mirror mode 
        if(mode === 'realtime'){
            this.context.save()
            this.context.scale(-1, 1)
            this.context.translate(-this.canvas.width, 0)
            // user display
            this.context.drawImage(user_video, 0,0, user_video.videoWidth*2, user_video.videoHeight, this.canvas.width/2, 0, this.canvas.width, this.canvas.height);
            this.context.restore()

        }else{
            // user display
            this.context.drawImage(user_video, 0,0, user_video.videoWidth*2, user_video.videoHeight, 0, 0, this.canvas.width, this.canvas.height);
        }

        // model diaplay
        this.context.drawImage(play_video, 0, 0, play_video.videoWidth*2, play_video.videoHeight, this.canvas.width/2, 0, this.canvas.width, this.canvas.height);

        //text
        this.context.font = `${0.00255*this.canvas.width}rem Roboto Slab, serif`;
        this.context.strokeStyle= '#1994af'
        this.context.strokeText('PLAY  DISPLAY', (this.canvas.width * 1.0155) /2, this.canvas.height * 0.105);
        this.context.fillText('PLAY  DISPLAY', (this.canvas.width * 1.0155) /2, this.canvas.height * 0.105);
        
        this.context.font = `${0.00255*this.canvas.width}rem Roboto Slab, serif`;
        this.context.strokeStyle= '#ffbb54'
        this.context.strokeText('USER  DISPLAY', this.canvas.width * 0.0155, this.canvas.height * 0.105);
        this.context.fillText('USER  DISPLAY', this.canvas.width * 0.0155, this.canvas.height * 0.105);
        
        this.context.font = `${0.00205*this.canvas.width}rem Roboto Slab, serif`;
        this.context.strokeText('Score', this.canvas.width * 0.0155, this.canvas.height * 0.805 );
        this.context.fillText('Score', this.canvas.width * 0.0155, this.canvas.height * 0.805 );
        
        this.context.font = `${0.00255*this.canvas.width}rem Roboto Slab, serif`;
        this.context.strokeText(total_score, this.canvas.width * 0.0155, this.canvas.height * 0.925);
        this.context.fillText(total_score, this.canvas.width * 0.0155, this.canvas.height * 0.925);

        
        
        this.loopID = window.requestAnimationFrame(timestamp=>{
            this.draw_play.bind(this, timestamp, mode, user_video, play_video, this.score)()
        })
    }
    draw_skeleton(){
        
        let image = new Image()
        image.width = this.skeletonCanvas.width
        image.height = this.skeletonCanvas.height
        image.onload = e => {
            this.skeletonContext.drawImage(image, image.width*0.1544, image.height*0.2173, image.width*1.222, image.height*1.3094, 0,0, this.skeletonCanvas.width, this.skeletonCanvas.height)
            let text = 'Skeleton Display'
            this.skeletonContext.font = `${0.00355*this.skeletonCanvas.width}rem Roboto Slab, serif`;
            this.skeletonContext.strokeStyle= '#ffbb54'
            this.skeletonContext.strokeText(text, (this.skeletonCanvas.width * 0.5155)/2, this.skeletonCanvas.height * 0.105);
            this.skeletonContext.fillText(text, (this.skeletonCanvas.width * 0.5155)/2, this.skeletonCanvas.height * 0.105);
        }
        image.src = "data:image/png;base64," + this.skeleton_image


    }
    updateScore(score) {
        this.diff += score - this.score
        this.count = this.count-1
        this.score = score
        this.context.font = `${0.00255*this.canvas.width}rem Roboto Slab, serif`;
        this.context.strokeStyle= '#ffbb54'
        this.context.strokeText(score, this.canvas.width * 0.0515, this.canvas.height * 0.925);
        this.context.fillText(score, this.canvas.width * 0.0515, this.canvas.height * 0.925);

        if(this.count==0){
            this.count=5
            let diff = this.diff/this.count
            if(diff > 4){
                this.color_frame.classList.remove('score-bad')
                this.color_frame.classList.remove('score-good')
                this.color_frame.classList.add('score-perfect')
            }else if(diff > 2.5){
                this.color_frame.classList.remove('score-bad')
                this.color_frame.classList.remove('score-perfect')
                this.color_frame.classList.add('score-good')
            }else{
                this.color_frame.classList.remove('score-good')
                this.color_frame.classList.remove('score-perfect')
                this.color_frame.classList.add('score-bad')
            }
            this.diff=0     
        }
    }
    endGame(){
        const stage_loader = document.getElementById('play-stage-loader')
        stage_loader.classList.remove('content-visible')
        stage_loader.classList.add('content-hide')
        this.context.font = `${0.0045*this.canvas.width}rem Roboto Slab, serif`;
        this.context.strokeStyle= '#a65bf8'
        this.context.strokeText('Play Done...', (this.canvas.width * 0.698) /2, (this.canvas.height * 0.798) /2);
        this.context.fillText('Play Done...', (this.canvas.width * 0.698) /2, (this.canvas.height * 0.798) /2);
    }
    
}

class PlayManager {
    constructor(navigator, msg_handler, play_view, pid, URL) {
        this.user_video = document.getElementById('userVideo'),
        this.play_video = document.getElementById('playVideo')
        this.mode = this.user_video.getAttribute('data-play-mode')
        this.rec = null
        this.fps = 30
        this.chunkLoopID = null
        this.total_score = 0
        this.parts_score = {
            '1_face_body' : 0,
            '2_right_arm' : 0,
            '3_right_leg' : 0,
            '4_left_leg' : 0,
            '5_left_arm' : 0,
        }
        this.msg_handler = msg_handler
        this.play_view = play_view
        this.config = {
            pid: pid,
            id: this.msg_handler.CLIENT_ID,
            start_date: new Date(Date.now()).toString()
        }
        this.URL = URL
        
        this.start_flag = false 
        this.preview_path = ''
        this.start_point = 0
        this.end_point = 0
        if(this.mode === 'realtime'){
            this.navigator = navigator
            this.navigator.getMedia = navigator.mediaDevices.getUserMedia || navigator.mediaDevices.webkitGetUserMedia || navigator.mediaDevices.mozGetuserMedia || navigator.mediaDevices.msGetUserMedia;
        }
    }
    main(){
        this.init()
    }
    load(){
        let play_container = document.getElementById('play-content') 
        let play_loader = document.getElementById('play-loader')
        let center_text = document.getElementById('center-text-neon')
        center_text.innerText = 'Done !'
        setTimeout((()=>{
            play_container.classList.remove('content-hide')
            play_container.classList.add('content-visible')
            play_loader.classList.remove('content-visible')
            play_loader.classList.add('content-hide')
            this.play_view.draw_intro(2, this.mode, this.user_video, this.play_video)
        }).bind(this), 1000)
        
    }

    init() {
        // resize 
        // window.onresize = this.resizeCanvas.bind(this)
        // this.play_view.resizeCanvas.bind(this.play_view)()

        this.play_view.setRecordCallback(this.record.bind(this))
        this.init_handler()

        this.play_video.pause()
        
        this.user_video.addEventListener('play', e => {
            console.log('play')
            if(this.start_flag){
                return
            }else{
                // this.user_video.pause()
                this.start_flag = true
            }
            
            if(this.mode === 'realtime'){
                console.log('realtime')
            }else if(this.mode === 'upload'){
                console.log('upload')
            }else{
                console.log('error')
            }

        }, false);

        // Start Play
        if(this.mode === 'realtime'){
            this.initCaptureVideo(this.user_video);
        }
        // Start PreProcessing
        if(this.mode === 'upload'){  
            this.user_video.pause() 
            let reg = /media\/(\S)+/;
            let user_src = reg.exec(this.user_video.src)[0]
            let play_src = reg.exec(this.play_video.src)[0]
            this.msg_handler.setOpenCallBack((()=>{
                this.msg_handler.sendObj('sync', {
                    'pid' : this.config.pid,
                    'user_video_src' : user_src,
                    'play_video_src' : play_src
                })
            }).bind(this))
        }
    }

    init_handler(){
        this.msg_handler.setReceiveCallBack(data=>{
            switch (data.type) {
                case 'message':
                    break;
                case 'sync':
                    console.log('sync')
                    console.log(data)
                    this.start_point = parseFloat(data.start) 
                    this.end_point = parseFloat(data.end)
                    this.play_video.currentTime = parseFloat(data.start)
                    this.load()
                    break
                case 'update_score':
                    this.total_score += data.score
                    this.parts_score['1_face_body'] += data.parts_score['face_body']
                    this.parts_score['2_right_arm'] += data.parts_score['right_arm']
                    this.parts_score['3_right_leg'] += data.parts_score['right_leg']            
                    this.parts_score['4_left_leg'] += data.parts_score['left_leg']
                    this.parts_score['5_left_arm'] += data.parts_score['left_arm']
                    this.play_view.skeleton_image = data.image
                    this.play_view.updateScore(this.total_score)
                    this.play_view.draw_skeleton()
                    break
                case 'update_preview_path':
                    this.preview_path = data.chunk_path
                    break
                case 'skeleton':
                    break
            }
        })
    }

    initCaptureVideo(video) {
        this.navigator.mediaDevices
            .getUserMedia({audio: false, video: true})
            .then(gotStream.bind(this))
            .catch(error => console.error(error));

        function gotStream(stream) {
            // video.src = window.URL.createObjectURL(stream);
            console.log(stream)
            this.user_video.srcObject = stream;
            this.user_video.play()
            setTimeout(this.load.bind(this), 2000)
        }
    }

    record() {
        const chunks = [];
        const stream = this.play_view.canvas.captureStream();
        this.rec = new MediaRecorder(stream);

        function send_chunk(){
            let recorder = new MediaRecorder(stream);
            let chunks = [];
            recorder.ondataavailable = e => {
                chunks.push(e.data)
            }
            recorder.onstop = e => {
                // this.msg_handler.send(data)
                
                if(this.play_video.currentTime < this.end_point || this.mode==='realtime'){
                    let data =new File(chunks,`${this.config.pid}_${e.timeStamp}.mp4`, {type: 'video/mp4'})
                    this.msg_handler.send(data)
                }
                // console.log(this.play_video.currentTime)
                if(this.play_video.currentTime >= this.end_point && this.mode==='upload'){
                    try{
                        this.rec.stop()
                    }catch(error){
                        console.log(error)
                    }finally{
                        this.play_video.pause()
                        this.user_video.pause()
                        clearInterval(this.msg_handler.CHUNK_LOOP_ID)
                    }
                }
            }
            setTimeout(()=> recorder.stop(), 1000); 
            recorder.start();
         }
                    

         const CHUNK_LOOP_ID = setInterval(send_chunk.bind(this), 1000) // have a 1s media file
         this.msg_handler.setChunkLoopID(CHUNK_LOOP_ID) 
          
        this.rec.addEventListener('dataavailable', e=>{
            chunks.push(e.data)
        })

        this.rec.addEventListener('stop', e=>{
            console.log('stopped')
            cancelAnimationFrame(this.play_view.loopID)
            clearInterval(this.msg_handler.CHUNK_LOOP_ID)
            this.msg_handler.close(this.config.pid)
            let video_title = this.user_video.getAttribute('data-title') + `_playvideo.mp4`
            this.exportVid(new Blob(chunks, {type: 'video/mp4'}), video_title)
            
            const play_data = new FormData()
            const file = new File(chunks, video_title, {'type':'video/mp4'})
            play_data.append('pid', this.config.pid)
            play_data.append('play_video', file, file.name)
            play_data.append('total_score', this.total_score)
            play_data.append('parts_score', JSON.stringify(this.parts_score))
            play_data.append('datetime', new Date(Date.now()).toString())
            play_data.append('preview_path', this.preview_path)
            this.msg_handler.sendResult(`http://${this.URL}/play/?pid=${this.config.pid}`, play_data)
            
            this.endGame() 
        })

        if(this.mode ==='realtime'){
            this.play_video.addEventListener('ended', e => {
                this.rec.stop()
            })
        }

        this.rec.start(2000);

    }
    exportVid(blob, title) {
        const video_preview = document.querySelector('#play-video-preview')     
        const line = document.querySelectorAll('.button-line')[0]
        const vid = document.createElement('video');
        const a = document.createElement('a');
        
        vid.src = URL.createObjectURL(blob);
        vid.controls = true;
        vid.classList.add('user_video')

        a.download = title;
        a.href = vid.src;
        
        const btn = document.createElement('button')
        btn.className = 'w-btn w-btn-gra1'
        btn.textContent = 'Download Play Video'

        a.appendChild(btn)
        line.appendChild(a);
        video_preview.appendChild(vid);

        line.classList.remove('content-hide')
        line.classList.add('content-visible')
        
        return vid
    }
    endGame(){
        this.play_view.endGame()
    
        const scoreText = document.getElementById('scoreText')
        scoreText.innerText = 'Score : ' + this.total_score 

        // polygon chart 
        // make integer score to under 0.
        let polygon_chart_el = document.getElementById("polygon-chart");
        console.log(this.parts_score)
        let parts_list = Object.values(this.parts_score)
        let max_num_length = 0
        let max_num = 0
        parts_list.forEach(score=>{
            let len = (score).toString().length
            if (len > max_num_length){
                max_num_length = len
                max_num = score
            }else if(len === max_num_length && score > max_num){
                max_num = score
            }
        })
        parts_list = parts_list.map(score=>{
            let len = max_num_length - (score).toString().length
            return parseFloat('0.' + '0'.repeat(len) + score)
        })
        parts_list.push(parts_list.shift())

        // play_result.js function
        create_polygon_chart(polygon_chart_el, [parts_list]).init()

        setTimeout(this.redirectToShare, 1500)
    }
    redirectToShare(){
        // result share section
        const share_page = document.querySelector('#result-share')
        const play_content = document.querySelector('#play-content')
  
        share_page.classList.remove('content-hide')
        share_page.classList.add('content-visible')
        play_content.classList.remove('content-visible')
        play_content.classList.add('content-hide')

        window.location = (""+window.location).replace(/#[A-Za-z0-9_]*$/,'')+"#result-share"        
    }
    
}

class MessageHandler {
    constructor(URL, pid=null) {
        this.CLIENT_ID = "client 1"
        this.URL = URL
        this.socket = new WebSocket(this.URL)
        this.receivedCallBack = ()=>{}
        this.openCallBack = ()=>{}
        this.CHUNK_LOOP_ID = null
        this.isOpen = false
        this.init()

    }
    init() {
        this.socket.onopen = e => {
            this.sendMessage('check', 'connected with client : ' + this.CLIENT_ID)
            this.isOpen = true
            this.openCallBack()
        }
        this.socket.onmessage = e => {
            // console.log(e)
            const data = JSON.parse(e.data)
            try{
                this.receivedCallBack(data)
            }catch(error){
                console.log(error)
            }
        }
        this.socket.onclose = e=>{
            console.log('disconnected')
            this.socket.close()
        }
    }
    send(data){
        this.socket.send(data)
    }
    sendObj(type, obj){
        this.socket.send(JSON.stringify({
            'type' : type,
            ...obj,
        }))
    }
    sendData(type, stamp, data){
        this.socket.send(JSON.stringify({
            'type' : type,
            'stamp' : stamp,
            'data': data,
        }))
    }
    sendMessage(type, msg) {
        this.socket.send(JSON.stringify({
            'type' : type,
            'message': msg,
        }))
    }
    setOpenCallBack(callback){
        this.openCallBack = callback
    }
    setReceiveCallBack(callback){
        this.receivedCallBack = callback
    }
    setChunkLoopID(loopID){
        this.CHUNK_LOOP_ID = loopID
    }
    close(pid=null) {
        this.socket.send(JSON.stringify({
            'type' : 'close',
            'pid' : pid,
            'message': 'close',
        }))
        this.socket.close()
    }
    makeMessage(func) {
        func()
    }
    async sendResult(URL, datas) {
        const request = new Request(URL + '', {
            headers: {
                'X-CSRFToken': this.getCookie("csrftoken")
            }
        })
        let res = await fetch(request, {
            method: 'POST',
            mode: 'same-origin',
            body: datas
        })
        return res
    }

    getCookie(name) {
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
}

function main() {
    const URL = '127.0.0.1:8000'
    const pid = document.querySelector('#userVideo').getAttribute('data-pid')
    const play_view = new PlayView()
    const msg_handler = new MessageHandler(`ws://${URL}/ws/play/${pid}`)
    const manager = new PlayManager(window.navigator, msg_handler, play_view, pid, URL)
    manager.main()

}
main()
