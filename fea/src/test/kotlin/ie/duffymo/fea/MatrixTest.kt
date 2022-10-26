package ie.duffymo.fea

import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Test
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import org.nd4j.linalg.api.ops.DynamicCustomOp
import org.nd4j.linalg.api.ops.impl.transforms.Cholesky
import org.nd4j.linalg.api.ops.impl.transforms.custom.Svd
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.factory.ops.NDLinalg

/**
 * @link https://deeplearning4j.konduit.ai/nd4j/how-to-guides/basics
 */
class MatrixTest {

    @Test
    fun `create and print a matrix`() {
        // setup
        val a = Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0, 4.0), intArrayOf(2, 2))
        // Values are loaded by rows
        val expected = """
[[    1.0000,    2.0000], 
 [    3.0000,    4.0000]]
         """.trimIndent()
        // exercise
        val s = a.toString()
        // assert
        Assertions.assertEquals(expected, s)
    }

    @Test
    fun `add a scalar to a matrix`() {
        // setup
        val a = Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0, 4.0), intArrayOf(2, 2))
        val expected = Nd4j.create(doubleArrayOf(2.0, 3.0, 4.0, 5.0), intArrayOf(2, 2))
        // exercise
        val actual = a.add(1)
        // assert
        Assertions.assertEquals(expected, actual)
    }

    @Test
    fun `add a scalar in-place to a matrix`() {
        // setup
        val a = Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0, 4.0), intArrayOf(2, 2))
        val expected = Nd4j.create(doubleArrayOf(2.0, 3.0, 4.0, 5.0), intArrayOf(2, 2))
        // exercise
        a.addi(1)
        // assert
        Assertions.assertEquals(expected, a)
    }

    @Test
    fun `subtract a scalar from a matrix`() {
        // setup
        val a = Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0, 4.0), intArrayOf(2, 2))
        val expected = Nd4j.create(doubleArrayOf(0.0, 1.0, 2.0, 3.0), intArrayOf(2, 2))
        // exercise
        val actual = a.sub(1)
        // assert
        Assertions.assertEquals(expected, actual)
    }

    @Test
    fun `subtract a scalar in-place from a matrix`() {
        // setup
        val a = Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0, 4.0), intArrayOf(2, 2))
        val expected = Nd4j.create(doubleArrayOf(0.0, 1.0, 2.0, 3.0), intArrayOf(2, 2))
        // exercise
        a.subi(1)
        // assert
        Assertions.assertEquals(expected, a)
    }

    @Test
    fun `multiply a matrix by a scalar`() {
        // setup
        val a = Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0, 4.0), intArrayOf(2, 2))
        val expected = Nd4j.create(doubleArrayOf(3.0, 6.0, 9.0, 12.0), intArrayOf(2, 2))
        // exercise
        val actual = a.mul(3)
        // assert
        Assertions.assertEquals(expected, actual)
    }

    @Test
    fun `multiply a matrix in-place by a scalar`() {
        // setup
        val a = Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0, 4.0), intArrayOf(2, 2))
        val expected = Nd4j.create(doubleArrayOf(3.0, 6.0, 9.0, 12.0), intArrayOf(2, 2))
        // exercise
        a.muli(3)
        // assert
        Assertions.assertEquals(expected, a)
    }


    @Test
    fun `divide a matrix by a scalar`() {
        // setup
        val a = Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0, 4.0), intArrayOf(2, 2))
        val expected = Nd4j.create(doubleArrayOf(0.5, 1.0, 1.5, 2.0), intArrayOf(2, 2))
        // exercise
        val actual = a.div(2)
        // assert
        Assertions.assertEquals(expected, actual)
    }

    @Test
    fun `divide a matrix in-place by a scalar`() {
        // setup
        val a = Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0, 4.0), intArrayOf(2, 2))
        val expected = Nd4j.create(doubleArrayOf(0.5, 1.0, 1.5, 2.0), intArrayOf(2, 2))
        // exercise
        a.divi(2)
        // assert
        Assertions.assertEquals(expected, a)
    }

    @Test
    fun `add a matrix and a row vector`() {
        // setup
        val a = Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0, 4.0), intArrayOf(2, 2))
        val row = Nd4j.create(doubleArrayOf(11.0, 13.0), intArrayOf(2))
        // Adds the row vector to all rows
        val expected = Nd4j.create(doubleArrayOf(12.0, 15.0, 14.0, 17.0), intArrayOf(2, 2))
        // exercise
        val actual = a.addRowVector(row)
        // assert
        Assertions.assertEquals(expected, actual)
    }

    @Test
    fun `add a matrix and a column vector`() {
        // setup
        val a = Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0, 4.0), intArrayOf(2, 2))
        val col = Nd4j.create(doubleArrayOf(11.0, 13.0), intArrayOf(2, 1))
        // Adds the column vector to all columns
        val expected = Nd4j.create(doubleArrayOf(12.0, 13.0, 16.0, 17.0), intArrayOf(2, 2))
        // exercise
        val actual = a.addColumnVector(col)
        // assert
        Assertions.assertEquals(expected, actual)
    }

    @Test
    fun `hadamard product of two matricies`() {
        // setup
        val a = Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0, 4.0), intArrayOf(2, 2))
        val b = Nd4j.create(doubleArrayOf(5.0, 6.0, 7.0, 8.0), intArrayOf(2, 2))
        // Adds the column vector to all columns
        val expected = Nd4j.create(doubleArrayOf(5.0, 12.0, 21.0, 32.0), intArrayOf(2, 2))
        // exercise
        val actual = a.mul(b)
        // assert
        Assertions.assertEquals(expected, actual)
    }

    @ParameterizedTest
    @ValueSource(longs = [0L, 1L])
    fun `add a row vector to each row in a matrix`(rowIndex : Long) {
        // setup
        val a = Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0, 4.0), intArrayOf(2, 2))
        val row = Nd4j.create(doubleArrayOf(11.0, 13.0), intArrayOf(2))
        val expected = arrayOf(
            Nd4j.create(doubleArrayOf(12.0, 15.0, 3.0, 4.0), intArrayOf(2, 2)),
            Nd4j.create(doubleArrayOf(1.0, 2.0, 14.0, 17.0), intArrayOf(2, 2)))
        // exercise
        // Adds the row vector to successive rows
        a.getRow(rowIndex).addi(row)
        // assert
        Assertions.assertEquals(expected[rowIndex.toInt()], a)
    }

    @ParameterizedTest
    @ValueSource(longs = [0L, 1L])
    fun `add a column vector to each column in a matrix`(colIndex : Long) {
        // setup
        val a = Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0, 4.0), intArrayOf(2, 2))
        val col = Nd4j.create(doubleArrayOf(11.0, 13.0), intArrayOf(2, 1))
        val expected = arrayOf(
            Nd4j.create(doubleArrayOf(12.0, 2.0, 16.0, 4.0), intArrayOf(2, 2)),
            Nd4j.create(doubleArrayOf(1.0, 13.0, 3.0, 17.0), intArrayOf(2, 2)))
        // exercise
        // Adds the column vector to successive columns
        a.getColumn(colIndex).reshape(intArrayOf(2, 1)).addi(col)
        // assert
        Assertions.assertEquals(expected[colIndex.toInt()], a)
    }

    @Test
    fun `add two matricies`() {
        // setup
        val a = Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0, 4.0), intArrayOf(2, 2))
        val b = Nd4j.create(doubleArrayOf(5.0, 6.0, 7.0, 8.0), intArrayOf(2, 2))
        val expected = Nd4j.create(doubleArrayOf(6.0, 8.0, 10.0, 12.0), intArrayOf(2, 2))
        // exercise
        val actual = a.add(b)
        // assert
        Assertions.assertEquals(expected, actual)
    }

    @Test
    fun `subtract two matricies`() {
        // setup
        val a = Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0, 4.0), intArrayOf(2, 2))
        val b = Nd4j.create(doubleArrayOf(5.0, 6.0, 7.0, 8.0), intArrayOf(2, 2))
        val expected = Nd4j.create(doubleArrayOf(-4.0, -4.0, -4.0, -4.0), intArrayOf(2, 2))
        // exercise
        val actual = a.sub(b)
        // assert
        Assertions.assertEquals(expected, actual)
    }

    @Test
    fun `multiply two matricies`() {
        // setup
        val a = Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0, 4.0), intArrayOf(2, 2))
        val b = Nd4j.create(doubleArrayOf(5.0, 6.0, 7.0, 8.0), intArrayOf(2, 2))
        // https://www.wolframalpha.com/input?i=matrix+multiplication+calculator&assumption=%7B%22F%22%2C+%22MatricesOperations%22%2C+%22theMatrix2%22%7D+-%3E%22%7B%7B5%2C+6%7D%2C+%7B7%2C8%7D%7D%22&assumption=%7B%22F%22%2C+%22MatricesOperations%22%2C+%22theMatrix1%22%7D+-%3E%22%7B%7B1%2C+2%7D%2C+%7B3%2C4%7D%7D%22
        val expected = Nd4j.create(doubleArrayOf(19.0, 22.0, 43.0, 50.0), intArrayOf(2, 2))
        // exercise
        val actual = a.mmul(b)
        // assert
        Assertions.assertEquals(expected, actual)
    }

    @Test
    fun `transpose a matrix`() {
        // setup
        val a = Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), intArrayOf(2, 3))
        val expected = Nd4j.create(doubleArrayOf(1.0, 4.0, 2.0, 5.0, 3.0, 6.0), intArrayOf(3, 2))
        // exercise
        val actual = a.transpose()
        // assert
        Assertions.assertEquals(expected, actual)
    }

    @Test
    fun `multiply a column vector by a matrix`() {
        // setup
        val a = Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0, 4.0), intArrayOf(2, 2))
        val col = Nd4j.create(doubleArrayOf(11.0, 13.0), intArrayOf(2, 1))
        val expected = Nd4j.create(doubleArrayOf(37.0, 85.0), intArrayOf(2, 1))
        // exercise
        val actual = a.mmul(col)
        // assert
        Assertions.assertEquals(expected, actual)
    }

    @Test
    fun `multiply a matrix by a row vector`() {
        val row = Nd4j.create(doubleArrayOf(11.0, 13.0), intArrayOf(1, 2))
        val a = Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0, 4.0), intArrayOf(2, 2))
        val expected = Nd4j.create(doubleArrayOf(50.0, 74.0), intArrayOf(1, 2))
        // exercise
        val actual = row.mmul(a)
        // assert
        Assertions.assertEquals(expected, actual)
    }

    @Test
    fun `lu decomposition`() {
        // setup
        // Strang "Intro To Applied Math" pages 8-11
        val a = Nd4j.create(doubleArrayOf(
            2.0, 1.0, 0.0, 0.0,
            1.0, 2.0, 1.0, 0.0,
            0.0, 1.0, 2.0, 1.0,
            0.0, 0.0, 1.0, 2.0,
        ), intArrayOf(4, 4))
        val expected = Nd4j.create(doubleArrayOf(
            2.0, 1.0, 0.0, 0.0,
            0.5, 1.5, 1.0, 0.0,
            0.0, 2.0/3.0, 4.0/3.0, 1.0,
            0.0, 0.0, 0.75, 1.25
        ), intArrayOf(4, 4))
        // exercise
        val actual = NDLinalg().lu(a)
        // assert
        Assertions.assertEquals(expected, actual)
    }

    @Test
    fun `solve a system of equations`() {
        // setup
        val a = Nd4j.create(doubleArrayOf(
            2.0, 1.0, 0.0, 0.0,
            1.0, 2.0, 1.0, 0.0,
            0.0, 1.0, 2.0, 1.0,
            0.0, 0.0, 1.0, 2.0,
        ), intArrayOf(4, 4))
        val b = Nd4j.create(doubleArrayOf(2.0, 1.0, 4.0, 8.0), intArrayOf(4, 1))
        val expected = Nd4j.create(doubleArrayOf(1.0, 0.0, 0.0, 4.0), intArrayOf(4, 1))
        // exercise
        val actual = NDLinalg().solve(a, b)
        // assert
        Assertions.assertEquals(expected, actual)
    }

    @Test
    fun `cholesky solution generation`() {
        // setup
        val a = Nd4j.create(doubleArrayOf(
            4.0, 12.0, -16.0,
            12.0, 37.0, -43.0,
            -16.0, -43.0, 98.0
        ), intArrayOf(3, 3))
        val b = Nd4j.create(doubleArrayOf(10.0, 20.0, 30.0), intArrayOf(3, 1))
        val expected = Nd4j.create(doubleArrayOf(285.8333, -76.6667, 13.3333), intArrayOf(3, 1))
        // exercise
        val actual = NDLinalg().solve(a, b)
        // assert
        Assertions.assertEquals(expected, actual)
    }

    @Test
    fun `qr decomposition`() {
        // setup
        // https://en.wikipedia.org/wiki/QR_decomposition
        val a = Nd4j.create(doubleArrayOf(
            12.0, -51.0, 4.0,
            6.0, 167.0, -68.0,
            -4.0, 24.0, -41.0,
        ), intArrayOf(3, 3))
        val q = Nd4j.create(doubleArrayOf(
            6.0/7.0, -69.0/175.0, 58.0/175.0,
            3.0/7.0, 158.0/175.0, -6.0/175.0,
            -2.0/7.0, 6.0/35.0, 33.0/35.0
        ), intArrayOf(3, 3))
        val r = Nd4j.create(doubleArrayOf(
            14.0, 21.0, -14.0,
            0.0, 175.0, -70.0,
            0.0, 0.0, -35.0
        ), intArrayOf(3, 3))
        // exercise
        val actual = NDLinalg().qr(a, true)
        // assert
        Assertions.assertEquals(q, actual[0], "No match for Q")
        Assertions.assertEquals(r, actual[1], "No match for R")
        Assertions.assertEquals(a, actual[0].mmul(actual[1]), "A should equal QR")
    }

    @Test
    @Disabled
    fun `singular value decomposition`() {
        // setup
        // https://stackoverflow.com/questions/19763698/solving-non-square-linear-system-with-r/19767525#19767525
        // https://stackoverflow.com/questions/74157832/runtime-error-from-nd4j-when-executing-svd/74175076#74175076
        Nd4j.getExecutioner().enableDebugMode(true)
        Nd4j.getExecutioner().enableVerboseMode(true)

        val a = Nd4j.create(doubleArrayOf(
            0.0, 1.0, -2.0, 3.0,
            5.0, -3.0, 1.0, -2.0,
            5.0, -2.0, -1.0, 1.0
        ), intArrayOf(3, 4))
        val b = Nd4j.create(doubleArrayOf(-17.0, 28.0, 11.0), intArrayOf(3, 1))
        val u = Nd4j.create(doubleArrayOf(
            -0.1295469, -0.8061540,  0.5773503,
            0.7629233,  0.2908861,  0.5773503,
            0.6333764, -0.5152679, -0.5773503
        ), intArrayOf(3, 3))
        val v = Nd4j.create(doubleArrayOf(
            0.87191556, -0.2515803, -0.1764323,
            -0.46022634, -0.1453716, -0.4694190,
            0.04853711,  0.5423235,  0.6394484,
            -0.15999723, -0.7883272,  0.5827720
        ), intArrayOf(3, 4))
        val d = Nd4j.create(doubleArrayOf(
            8.007081e+00, 4.459446e+00, 4.022656e-16
        ), intArrayOf(3))
        // exercise
        val actual = DynamicCustomOp.builder("svd")
            .addInputs(a)
            .addIntegerArguments(1, 1, Svd.DEFAULT_SWITCHNUM)
            .build()
        Nd4j.linalg().svd(a, true, true)
        // assert
        Assertions.assertTrue(true)
    }

    @Test
    fun `cholesky decomposition`() {
        // setup
        // https://en.wikipedia.org/wiki/Cholesky_decomposition
        // Note: this matrix is symmetric.  Is it positive definite?  Required for Cholesky
        val a = Nd4j.create(doubleArrayOf(
            4.0, 12.0, -16.0,
            12.0, 37.0, -43.0,
            -16.0, -43.0, 98.0
        ), intArrayOf(3, 3))
        val b = Nd4j.create(doubleArrayOf(10.0, 20.0, 30.0), intArrayOf(3, 1))
        val x = Nd4j.create(doubleArrayOf(285.8333, -76.6667, 13.3333), intArrayOf(3, 1))
        val l = Nd4j.create(doubleArrayOf(
            2.0, 0.0, 0.0,
            6.0, 1.0, 0.0,
            -8.0, 5.0, 3.0
        ), intArrayOf(3, 3))
        // exercise
        val cholesky = Cholesky(a)
        val actual = Nd4j.getExecutioner().exec(cholesky).toList()[0]
        // assert
        Assertions.assertEquals(l, actual)
        Assertions.assertEquals(a, actual.mmul(actual.transpose()))
        // forward-back substitution
        val y = NDLinalg().triangularSolve(actual, b, true, false)
        println(y)
        val z = NDLinalg().triangularSolve(actual.transpose(), y, false, false)
        Assertions.assertEquals(x, z)
    }

    @Test
    fun `create a matrix of zeroes`() {
        // setup
        val a = Nd4j.zeros(10, 10)
        val expected = """
[[         0,         0,         0,         0,         0,         0,         0,         0,         0,         0], 
 [         0,         0,         0,         0,         0,         0,         0,         0,         0,         0], 
 [         0,         0,         0,         0,         0,         0,         0,         0,         0,         0], 
 [         0,         0,         0,         0,         0,         0,         0,         0,         0,         0], 
 [         0,         0,         0,         0,         0,         0,         0,         0,         0,         0], 
 [         0,         0,         0,         0,         0,         0,         0,         0,         0,         0], 
 [         0,         0,         0,         0,         0,         0,         0,         0,         0,         0], 
 [         0,         0,         0,         0,         0,         0,         0,         0,         0,         0], 
 [         0,         0,         0,         0,         0,         0,         0,         0,         0,         0], 
 [         0,         0,         0,         0,         0,         0,         0,         0,         0,         0]]""".trimIndent()
        // exercise
        // assert
        Assertions.assertEquals(expected, a.toString())
    }

    @Test
    fun `identity matrix()`() {
        // setup
        val identity = Nd4j.eye(10)
        val expected = """
[[    1.0000,         0,         0,         0,         0,         0,         0,         0,         0,         0], 
 [         0,    1.0000,         0,         0,         0,         0,         0,         0,         0,         0], 
 [         0,         0,    1.0000,         0,         0,         0,         0,         0,         0,         0], 
 [         0,         0,         0,    1.0000,         0,         0,         0,         0,         0,         0], 
 [         0,         0,         0,         0,    1.0000,         0,         0,         0,         0,         0], 
 [         0,         0,         0,         0,         0,    1.0000,         0,         0,         0,         0], 
 [         0,         0,         0,         0,         0,         0,    1.0000,         0,         0,         0], 
 [         0,         0,         0,         0,         0,         0,         0,    1.0000,         0,         0], 
 [         0,         0,         0,         0,         0,         0,         0,         0,    1.0000,         0], 
 [         0,         0,         0,         0,         0,         0,         0,         0,         0,    1.0000]]""".trimIndent()
        // exercise
        // assert
        Assertions.assertEquals(expected, identity.toString())
    }
}
