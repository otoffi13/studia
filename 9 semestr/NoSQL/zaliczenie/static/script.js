let pollId = null;
let pollChart = null;
const baseUrl = "http://127.0.0.1:5000";


function showHome() {
    document.getElementById("home-container").style.display = "block";
    document.getElementById("poll-container").style.display = "none";
    document.getElementById("create-poll-container").style.display = "none";
}

function showCreatePoll() {
    document.getElementById("home-container").style.display = "none";
    document.getElementById("create-poll-container").style.display = "block";
    document.getElementById("poll-container").style.display = "none";
}

function showPoll() {
    document.getElementById("home-container").style.display = "none";
    document.getElementById("poll-container").style.display = "block";
    document.getElementById("create-poll-container").style.display = "none";
}

async function loadPollsList() {
    const res = await fetch(`${baseUrl}/polls`);
    const polls = await res.json();

    const container = document.getElementById("polls-list");
    container.innerHTML = "";

    polls.forEach(poll => {
        const div = document.createElement("div");
        div.className = "poll-item";
        div.innerHTML = `
            <b>${poll.question}</b> <br>
            Status: ${poll.status}, kończy się: ${poll.end_time} <br>
            <button ${poll.status === "ended" ? "" : ""} onclick="selectPoll('${poll.id}')">
                ${poll.status === "ended" ? "Zakończona – zobacz wyniki" : "Głosuj"}
            </button>
        `;
        container.appendChild(div);
    });
}

function selectPoll(id) {
    pollId = id;
    loadPoll();
    showPoll();
}

async function loadPoll() {
    if (!pollId) return;

    const res = await fetch(`${baseUrl}/results/${pollId}`);
    const data = await res.json();

    document.getElementById("question").textContent = data.question;

    const maxChoices = parseInt(data.max_choices);
    const type = maxChoices === 1 ? "radio" : "checkbox";

    const voteForm = document.getElementById("vote-form");
    const optionsDiv = document.getElementById("options");
    optionsDiv.innerHTML = "";

    const now = Math.floor(Date.now() / 1000);
    const pollEnded = now > parseInt(data.end_time);

    voteForm.style.display = pollEnded ? "none" : "block";

    for (const option of Object.keys(data.results)) {
        const div = document.createElement("div");
        div.className = "option";

        const input = document.createElement("input");
        input.type = type;
        input.name = "option";
        input.value = option;
        input.disabled = pollEnded;

        const label = document.createElement("label");
        label.textContent = option;

        div.appendChild(input);
        div.appendChild(label);
        optionsDiv.appendChild(div);
    }

    updateResults(data.results);
}


document.getElementById("vote-form").addEventListener("submit", async (e) => {
    e.preventDefault();

    if (!pollId) {
        alert("Wybierz ankietę!");
        return;
    }

    const selected = Array.from(document.querySelectorAll("input[name='option']:checked"))
                         .map(el => el.value);

    if (selected.length === 0) {
        alert("Wybierz przynajmniej jedną opcję");
        return;
    }

    const res = await fetch(`${baseUrl}/vote`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ poll_id: pollId, options: selected })
    });

    const data = await res.json();

    if (res.status === 200) {
        alert("Głos oddany!");
        loadPoll();
        loadPollsList();
    } else {
        alert("Błąd: " + data.error);
    }
});


function updateResults(results) {
    const list = document.getElementById("results-list");
    list.innerHTML = "";

    const labels = [];
    const dataValues = [];

    for (const [option, count] of Object.entries(results)) {
        const li = document.createElement("li");
        li.textContent = `${option}: ${count} głosów`;
        list.appendChild(li);

        labels.push(option);
        dataValues.push(parseInt(count));
    }

    drawChart(labels, dataValues);
}

function drawChart(labels, data) {
    const ctx = document.getElementById("results-chart").getContext("2d");

    if (pollChart) {
        pollChart.destroy();
    }

    pollChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                label: 'Głosy',
                data: data,
                backgroundColor: [
                    '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'
                ]
            }]
        },
        options: {
            responsive: true
        }
    });
}

document.getElementById("create-poll-form").addEventListener("submit", async (e) => {
    e.preventDefault();

    const question = document.getElementById("new-poll-question").value;
    const options = document.getElementById("new-poll-options").value.split(",").map(s => s.trim());
    const max_choices = parseInt(document.getElementById("new-poll-max").value);

    const end_dt = document.getElementById("new-poll-end").value;
    const end_time = Math.floor(new Date(end_dt).getTime() / 1000);

    const res = await fetch(`${baseUrl}/poll`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, options, max_choices, end_time })
    });

    const data = await res.json();

    if (res.status === 201) {
        alert(`Ankieta utworzona! ID: ${data.id}`);
        showHome();
        loadPollsList();
    } else {
        alert("Błąd przy tworzeniu ankiety: " + data.error);
    }
});

//co 5 sekund
setInterval(loadPollsList, 5000);
loadPollsList();
