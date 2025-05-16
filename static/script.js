const stockContainer = document.getElementById("stock-container");
const infoContainer = document.getElementById("info-container");
const filterButton = document.querySelector(".filter-button");
const stockName = document.getElementById("stock-name-container");
const investmentScore = document.getElementById("investment-score-container");
const stockRisk = document.getElementById("risk-container");
const rsiPenalty = document.getElementById("rsi-penalty-container");
const meanReturn = document.getElementById("mean-return-container");

let stocks = [];

function display(rank, stock, risk, penalty, mean, score) {
    infoContainer.style.display = 'flex';
    stockName.innerHTML = `<strong>${stock}</strong>`;
    investmentScore.innerHTML = "Invesment Score: " + score;
    stockRisk.innerHTML = "Risk: " + (parseFloat(risk) * 100).toFixed(2) + "%";
    rsiPenalty.innerHTML = "RSI Penalty: " + penalty;
    meanReturn.innerHTML = "Mean Return: " + mean;

    if (parseInt(rank) < 4) {
        changeColor('rgb(53, 141, 52)');
    } else if (parseInt(rank) > 3 && parseInt(rank) < 6) {
        changeColor('rgb(208, 95, 53)');
    } else if (parseInt(rank) > 5) {
        changeColor('rgb(148, 37, 37)');
    }
}

function changeColor(color) {
    stockName.style.backgroundColor = color;
    investmentScore.style.backgroundColor = color;
    stockRisk.style.backgroundColor = color;
    rsiPenalty.style.backgroundColor = color;
    meanReturn.style.backgroundColor = color;
}

async function fetchStockRankings() {
    try {
        const response = await fetch("/rankings");
        const data = await response.json();

        if (data.error) {
            stockContainer.innerHTML = '<label>STOCK DATA UNAVAILABLE.</label>';
            return;
        }
        Object.entries(data).forEach(([stockSymbol, stockArray]) =>{
            const stockData = stockArray[0];
            let values = [stockData['rank'], stockData['stock'], stockData['price'], stockData['change'], stockData['change percent'], stockData['volume'], stockData['predicted price'], stockData['risk'], stockData['rsi penalty'], stockData['mean return'], stockData['investment score']];
            stocks.push(values);
        })
        ascending();

    } catch (error) {
        stockContainer.innerHTML = '<label>ERROR FETCHING DATA.</label>';
        console.error("Error:", error);
    }   
}

$(function () {
    let filterSwitch = false;
    $(".filter-button").on('click', function () {
        if (filterSwitch) {
            filterButton.textContent = "ASCENDING";
            ascending();
        } else {
            filterButton.textContent = "DESCENDING";
            descending();
        }
        filterSwitch = !filterSwitch;
    })

    window.onload = fetchStockRankings();
})

const ascending = () => {
    let temp = [];
    for (let i = 0; i < 7; i++) {
        for (let j = 0; j < 7 - 1; j++) {
            if (stocks[j][0] > stocks[j + 1][0]) {
                temp = stocks[j];
                stocks[j] = stocks[j + 1];
                stocks[j + 1] = temp;
            }
        }
    }

    displayRankings();
}

function displayRankings() {
    stockContainer.innerHTML = "";
    for (let i = 0; i < 7; i++) {
        stockContainer.innerHTML += `
         <tr onclick="display('${stocks[i][0]}','${stocks[i][1]}','${stocks[i][7]}','${stocks[i][8]}','${stocks[i][9]}','${stocks[i][10]}')" id="${stocks[i][1]}">
            <td>${stocks[i][0]}</td>   
            <td>${stocks[i][1]}</td>
            <td>${stocks[i][2]}</td>
            <td id="${stocks[i][3]}">${stocks[i][3]}</td>
            <td id="${stocks[i][4]}">${stocks[i][4]}</td>   
            <td>${stocks[i][5]}</td>   
            <td>${stocks[i][6]}</td>                   
        </tr>
        `;

        if(parseFloat(stocks[i][3]) < 0){
            document.getElementById(`${stocks[i][3]}`).style.color = 'rgb(229, 62, 62)';
            document.getElementById(`${stocks[i][4]}`).style.color = 'rgb(229, 62, 62)';
        }else{
            document.getElementById(`${stocks[i][3]}`).style.color = 'rgb(90, 229, 62)';
            document.getElementById(`${stocks[i][4]}`).style.color = 'rgb(90, 229, 62)';
        }
    }
}

const descending = () => {
    let temp = [];
    for (let i = 0; i < 7; i++) {
        for (let j = 0; j < 7 - 1; j++) {
            if (stocks[j][0] < stocks[j + 1][0]) {
                temp = stocks[j];
                stocks[j] = stocks[j + 1];
                stocks[j + 1] = temp;
            }
        }
    }

    displayRankings();
}

function refresh() {
    stockContainer.innerHTML = "";
    fetchStockRankings();
    changeColor('rgb(233, 233, 233)');
    infoContainer.style.display = 'none';
}

function displayDescription() {

}

async function retrieveData(){
    // let data = {};

    // for(let i = 0; i < stocks.length; i++){
    //     data[String(stocks[i][1])] = new Set();

    //     let vals = [];
    //     for(let j = 0; j < stocks[i].length; j++){
    //         if(j === 1)
    //             continue;

    //         vals.push(stocks[i][j]);
    //     }
    //     data[String(stocks[i][1])] = vals;
    // }

    const response = await fetch("/rankings");
    const datas = await response.json();
    console.log(datas)
}

// // ================== CHAT FUNCTIONS ==================
// function appendMessage(text, sender) {
//     const messagesDiv = document.querySelector('.chat-messages');
//     const messageDiv = document.createElement('div');
//     messageDiv.className = `message ${sender}-message`;
//     messageDiv.textContent = text;
//     messagesDiv.appendChild(messageDiv);
    
//     // Auto-scroll to bottom
//     messagesDiv.scrollTop = messagesDiv.scrollHeight;
// }

// async function sendMessage() {
//     const input = document.querySelector('.message-input');
//     const message = input.value.trim();
    
//     if (!message) return;

//     appendMessage(message, 'user');
//     input.value = '';
    
//     try {
//         const response = await fetch('/chat', {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json'  // This header is crucial
//             },
//             body: JSON.stringify({ 
//                 message: message 
//             })
//         });
        
//         const data = await response.json();
//         appendMessage(data.response || data.error, 'bot');
        
//     } catch (error) {
//         appendMessage("Connection error. Please try again.", 'bot');
//     }
// }

// document.querySelector('.send-message').addEventListener('click', sendMessage);
// document.querySelector('.message-input').addEventListener('keypress', (e) => {
//     if (e.key === 'Enter') sendMessage();
// });

// // Initialize after DOM loads
// window.addEventListener('DOMContentLoaded', () => {
//     fetchStockRankings();
//     // Remove dummy messages
//     document.querySelectorAll('.chat-messages .message').forEach(el => el.remove());
// });