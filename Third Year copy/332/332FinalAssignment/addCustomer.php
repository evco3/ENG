<!DOCTYPE html>
<html>
<head>
	<title>Delicious Restaurant</title>
	<meta charset="UTF-8">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<style>
		body {
			font-family: Arial, sans-serif;
			margin: 0;
			padding: 0;
		}
		header {
			background-color: #3e2609;
			padding: 10px;
			text-align: center;
		}
		h1 {
			margin: 0;
			font-size: 36px;
			color: #fff;
		}
        h2 {
			margin: 0;
			font-size: 36px;
			color: #333;
            text-align: center;
		}
		nav {
			background-color: #333;
			color: #fff;
			display: flex;
			justify-content: space-around;
			align-items: center;
			padding: 10px;
		}
		nav a {
			color: #fff;
			text-decoration: none;
			font-size: 20px;
		}
		nav a:hover {
			color: #09ceff;
		}
		section {
			display: flex;
			flex-wrap: wrap;
			padding: 20px;
			text-align: center;
		}
		article {
			flex-basis: 400px;
			margin: 20px;
			background-color: #fff;
			box-shadow: 2px 2px 5px #ccc;
			border-radius: 5px;
			overflow: hidden;
		}
		article img {
			width: 100%;
			height: 200px;
			object-fit: cover;
		}
		article h2 {
			margin: 10px;
			font-size: 24px;
			color: #333;
		}
		article p {
			margin: 10px;
			font-size: 18px;
			color: #777;
			line-height: 1.5;
		}
		article button {
			display: block;
			margin: 10px auto;
			padding: 10px 20px;
			background-color: #f2b632;
			color: #fff;
			border: none;
			border-radius: 20px;
			font-size: 18px;
			cursor: pointer;
		}
		article button:hover {
			background-color: #333;
		}
        form{
            text-align: center;
        }
		footer {
			background-color: #333;
			color: #fff;
			text-align: center;
			padding: 10px;
		}
		footer p {
			margin: 0;
			font-size: 16px;
		}
		
	</style>
</head>
<body>
	<header>
		<h1>Delicious Restaurant</h1>
	</header>
	<nav>
		<a href="restaurant.html">Home</a>
		<a href="empSchedule.php">Schedule</a>
		<a href="Orders.php">Orders</a>
		<a href="#">Add Customer</a>
        <a href="orderSum.php">Order Summary</a>
	</nav>

<body>
    <h2>Add New Customer</h2>
    <form action="<?php echo $_SERVER['PHP_SELF'];?>" method="POST">
        <label for="firstName">First Name:</label>
        <input type="text" id="firstName" name="firstName" required><br><br>

        <label for="lastName">Last Name:</label>
        <input type="text" id="lastName" name="lastName" required><br><br>

        <label for="phoneNum">Phone Number:</label>
        <input type="text" id="phoneNum" name="phoneNum" required><br><br>

        <label for="city">City:</label>
        <input type="text" id="city" name="city" required><br><br>

        <label for="street">Street:</label>
        <input type="text" id="street" name="street" required><br><br>

        <label for="pCode">Postal Code:</label>
        <input type="text" id="pCode" name="pCode" required><br><br>

        <label for="email">Email:</label>
        <input type="email" id="email" name="email" required><br><br>

        <input type="submit" name="submit" value="Submit" >
    </form>
	<?php



	if(isset($_POST['submit'])){
		require_once 'connectDB.php';


		$firstName = $_POST["firstName"];
		$lastName = $_POST["lastName"];
		$phoneNum = $_POST["phoneNum"];
		$city = $_POST["city"];
		$street = $_POST["street"];
		$pCode = $_POST["pCode"];
		$email = $_POST["email"];


		// Check if customer already exists
		$sql = "SELECT * FROM customer WHERE email = ?";
		$result = $conn->prepare($sql);
		$result->execute([$email]);


		if ($result->rowCount() == 0) {
			// Insert new customer with $5.00 credit
			$sql2 = "INSERT INTO customer(firstName, lastName, credit, phoneNum, city, street, pCode, email)
				VALUES (?,?,5.00,?,?,?,?,?)";
			$result2 = $conn->prepare($sql2);
			$result2->execute([$firstName, $lastName, $phoneNum, $city, $street, $pCode, $email]);

			echo "<script>alert('The customer was succesfully added');</script>";

		} else {
			echo"<section>Customer already exists.</section>";
		}

		$conn->close();
	}
	?>
</body>

</html>
