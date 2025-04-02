const stockContainer = document.querySelector(".stock-container");
const infoContainer = document.getElementById("info-container");
const stock = document.getElementById("stock");
const stockName = document.getElementById("stock-name");

function display(){
    infoContainer.innerHTML = stockName.textContent;
    console.log("Im pressed.");
}

// $(function(){s
//     $(".stock").on('click',function(){
//         $(".stock").hide();
//     })
// })


// })
// function refresh(){

// }

// function arrange(){

// }

// function displayDescription(){

// }


// //this feature will be added later
// function chat(){

// }