function initTime() {
    let now = new Date();
    let hours = now.getHours();
    let minutes = now.getMinutes();
    let timeContainer = document.getElementById("timeh1")

    hours = hours < 10 ? '0' + hours : hours;
    minutes = minutes < 10 ? '0' + minutes : minutes;

    timeContainer.innerText = hours + ":" + minutes
    setTimeout(initTime, 100);
}
initTime()