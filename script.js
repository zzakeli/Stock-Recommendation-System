const stockContainer = document.querySelector(".stock-container");
const infoContainer = document.getElementById("info-container");
const stock = document.querySelector("stock");
const stockName = document.getElementById("stock-name");

$(function(){
    $(".stock").on('click',function(){
        infoContainer.innerHTML = this.innerHTML;
    })
})


function refresh(){

}

function arrange(){

}

function displayDescription(){

}


//this feature will be added later
function chat(){

}