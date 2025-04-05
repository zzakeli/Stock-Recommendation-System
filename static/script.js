const stockContainer = document.querySelector(".stock-container");
const infoContainer = document.getElementById("info-container");
const filterButton = document.querySelector(".filter-button");
// const stockName = document.getElementById("stock-name");


function display(stockName){
    const myStock = document.getElementById(stockName);
    infoContainer.innerHTML = myStock.textContent;
}

async function fetchStockRankings(){
    try {
        const response = await fetch("/rankings");
        const data = await response.json();
        
        if(data.error){
            stockContainer.innerHTML = '<label>STOCK DATA UNAVAILABLE.</label>';
            return;
        }

        data.rankings.forEach((rank) => {
            stockContainer.innerHTML += `<div onclick="display('${rank.stock}')" class="stock" id="${rank.stock}">${rank.rank}, ${rank.stock}, ${rank["Investment Score"]}</div>`;
        });

    } catch (error) {
        stockContainer.innerHTML = '<label>ERROR FETCHING DATA.</label>';
        console.error("Error:", error);
    }
}

$(function(){
    // let stocks = ["APPL","NVDA","AMZN","TSLA","MSFT","META","GOOGL"];
    // for(let i = 0; i < stocks.length; i++){
    //     stockContainer.innerHTML += `<div onclick="display('${stocks[i]}')" class="stock" id="${stocks[i]}">${stocks[i]}</div>`;
    // }
    // $(".stock").on('click',function(){
    //     infoContainer.innerHTML = this.innerHTML;
    // })
    let filterSwitch = true;
    $(".filter-button").on('click',function(){
        if(filterSwitch){
            filterButton.textContent = "ASCENDING";
            ascending();
        }else{
            filterButton.textContent = "DESCENDING";
            descending();
        }
        filterSwitch = !filterSwitch;   
    })

    window.onload =  fetchStockRankings();
})

const ascending = () =>{

}
const descending= () =>{

}

function refresh(){
    stockContainer.innerHTML = "";
    fetchStockRankings();
}


function arrange(){

}

function displayDescription(){

}


//this feature will be added later
function chat(){

}