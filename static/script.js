const stockContainer = document.getElementById("stock-container");
const infoContainer = document.getElementById("info-container");
const filterButton = document.querySelector(".filter-button");
const stockName = document.getElementById("stock-name-container");
const investmentScore = document.getElementById("investment-score-container");
const stockRisk = document.getElementById("risk-container");
const rsiPenalty = document.getElementById("rsi-penalty-container");
const meanReturn = document.getElementById("mean-return-container");

function display(rank,stock,risk,penalty,mean,score){
    infoContainer.style.display = 'flex';
    stockName.innerHTML = `<strong>${stock}</strong>`;
    investmentScore.innerHTML = "Invesment Score: " + score;
    stockRisk.innerHTML = "Risk: " +  (parseFloat(risk) * 100).toFixed(2) + "%";
    rsiPenalty.innerHTML = "RSI Penalty: " + penalty;
    meanReturn.innerHTML = "Mean Return: " + mean;
    
    if(parseInt(rank) < 4){
        changeColor('rgb(53, 141, 52)');
    }else if(parseInt(rank) > 3 && parseInt(rank) < 6){
        changeColor('rgb(208, 95, 53)');
    }else if(parseInt(rank) > 5){
        changeColor('rgb(148, 37, 37)');
    }
}

function changeColor(color){
    stockName.style.backgroundColor = color;
    investmentScore.style.backgroundColor = color;
    stockRisk.style.backgroundColor = color;
    rsiPenalty.style.backgroundColor = color;
    meanReturn.style.backgroundColor = color;
}

async function fetchStockRankings(){
    try {
        const response = await fetch("/rankings");
        const data = await response.json();
        
        if(data.error){
            stockContainer.innerHTML = '<label>STOCK DATA UNAVAILABLE.</label>';
            return;
        }

        data.rankings.forEach(rank => {
            // stockContainer.innerHTML += `<div onclick="display('${rank.stock}')" class="stock" id="${rank.stock}">${rank["rank"]}, ${rank["symbol"]}, ${rank["investment_score"]}, ${rank["current_price"]}, ${rank["change"]}, ${rank["change_percent"]}, ${rank["predicted_price"]}</div>`;
            stockContainer.innerHTML += `
            <tr onclick="display('${rank['rank']}','${rank['stock']}','${rank['risk']}','${rank['rsi penalty']}','${rank['mean return']}','${rank['investment score']}')" id="${rank['stock']}">
                <td>${rank['rank']}</td>   
                <td>${rank['stock']}</td>
                <td>${rank['price']}</td>
                <td>${rank['change']}</td>
                <td>${rank['change percent']}</td>   
                <td>${rank['volume']}</td>   
                <td>${rank['predicted price']}</td>                   
            </tr>`;
            // stockContainer.innerHTML += `<div onclick="display()" class="stock" id="">${rank['stock']},${rank['Investment Score']}, ${rank['Prediction']}</div>`;
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
    changeColor('rgb(233, 233, 233)');
    infoContainer.style.display = 'none';
}


function arrange(){

}

function displayDescription(){

}


//this feature will be added later
function chat(){

}