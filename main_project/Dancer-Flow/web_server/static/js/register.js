const username = document.querySelector('#username');
const feedBackArea = document.querySelector('.invalid-feedback');
const email = document.querySelector('#email');
const emailFeedBackArea = document.querySelector(".emailFeedBackArea");

const pw = document.querySelector('#password')
const repw = document.querySelector('#re_password')
const passwordFeedBackArea = document.querySelector(".passwordFeedBackArea");
const repasswordFeedBackArea = document.querySelector(".repasswordFeedBackArea");

const submit_btn = document.querySelector('#submit-btn')


pw.addEventListener('blur', e=>{
  if (repw.value !== '' &&  pw.value !== repw.value){
    pw.classList.add("is-invalid");
    passwordFeedBackArea.style.display = "block";
    passwordFeedBackArea.innerHTML = `<p>Password Missmatch</p>`;
    submit_btn.disabled = true
  }else{
    pw.classList.remove("is-invalid");
    passwordFeedBackArea.style.display = "none";
    submit_btn.disabled = false
  }
  repw.blur()
})

repw.addEventListener('blur', e=>{
  if (pw.value !== repw.value){
    repw.classList.add("is-invalid");
    repasswordFeedBackArea.style.display = "block";
    repasswordFeedBackArea.innerHTML = `<p>Password Missmatch</p>`;
    submit_btn.disabled = true
  }else{
    repw.classList.remove("is-invalid");
    repasswordFeedBackArea.style.display = "none";
    submit_btn.disabled = false
  }
  pw.blur()
})


// 사용자가 email blcok안에 키보드 입력을 감지.
email.addEventListener("blur", (e) => {
  const emailVal = e.target.value;

  email.classList.remove("is-invalid");
  emailFeedBackArea.style.display = "none";
  submit_btn.disabled = false

  if (emailVal.length > 0) {
    fetch("/user/validate-email", {
      body: JSON.stringify({ email: emailVal }), //JSON 문자열로 변환할 값
      method: "POST",
    })
      .then((res) => res.json())
      .then((data) => {
        console.log("data", data);
        if (data.email_error) {
          email.classList.add("is-invalid");
          emailFeedBackArea.style.display = "block";
          emailFeedBackArea.innerHTML = `<p>${data.email_error}</p>`;
          submit_btn.disabled = true

        }
        else{
          submit_btn.disabled = false

        }
      });
  }
});




// 사용자가 username block안에 키보드 입력을 감지. 
username.addEventListener('blur', (e) => {
  const usernameVal = e.target.value;

  username.classList.remove("is-invalid");
  feedBackArea.style.display = "none";
  submit_btn.disabled = false

  if (usernameVal.length > 0) {
    fetch("/user/validate-username", {
      body: JSON.stringify({username: usernameVal }), //JSON 문자열로 변환할 값
      method:"POST",
    })
      .then((res) => res.json())
      .then((data) => {
        console.log("data", data);
        if (data.username_error){
          username.classList.add("is-invalid");
          //get error from data and display in in feedBackArea
          feedBackArea.style.display ="block";
          feedBackArea.innerHTML=`<p>${data.username_error}</p>`;
          submit_btn.disabled = true

        }else{
          submit_btn.disabled = false

        }
      });
  }
});