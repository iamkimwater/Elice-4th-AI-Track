// console.log("1");
// console.log("2");

// function test(callback) {
// 	setTimeout (function() {
// 		console.log("3");
// 	}, 2000);
// }

// function print4() {
// 	console.log("4");
// }

// test(print4);










/*
// 콜백 대신 Promise 써보기

// callback hell

// id와 password를 받아서 로그인을 시도하고,
// 로그인이 성공한다면 로그인한 유저의 역할을 받아오는 프로그램
// db를 받아온다고 가정하고 db랑 비교함


// let id;         // 입력받을 id
// let password;  // 입력받을 pw

function loginUser(id, password, onSuccess, onError) {
	setTimeout(() => {
		const userDB = [
			{
				id: "paul",
				pw: "pw1",
			},
			{
				id: "tom",
				pw: "pw2",
			},
			{
				id: "grey",
				pw: "pw3",
			},
		];

		// console.log(userDB);
		for(let i=0; i<userDB.length; i++) {
			if (
				userDB[i].id === id &&
				userDB[i].pw === password
			) {
				onSuccess(id);
				return;
			}
		}
	}, 2000); // db에서 받아오는 시간이 2초가 걸린다고 가정
}

let id = 'paul';
let password = 'pw1';

loginUser(
	id,
	password,
	function (user) {
		console.log(user);
		console.log('로그인 성공!')
	},
	function (user) {
		console.log('아이디와 패스워드가 맞지 않습니다.')
	}
)

*/





/*

// promise

// promise 는 3가지 상태가 있다.
// pending(준비), fulfilled(이행), rejected(거부)

const myPromise = new Promise((resolve, reject) => {
	reject('프로미스가 거부됨')
});

myPromise.then((value) =>  {
	console.log(value);
});

*/



