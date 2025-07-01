#include "NDArray.h"
#include "Test.h"


using namespace std;


namespace
{
	void Test_Add()
	{
		// Add(matrix,scalar)
		{
			NDArray x = NDData::New({3,2},
				{
					11,21,
					31,12,
					22,32
				});
			NDArray y = x+0.5;
			Assert(y.IsEqualTo(NDData::New({3,2},
				{
					11.5,21.5,
					31.5,12.5,
					22.5,32.5
				})),"Add(matrix,scalar).");
		}

		// Add(matrix,column)
		{
			NDArray x = NDData::New({3,2},
				{
					11,12,
					21,22,
					31,32
				});
			print(x);
			NDArray y = NDData::New({2},{10,20});
			print(y);
			NDArray z = x+y;
			print(z);
			Assert(z.IsEqualTo(NDData::New({3,2},
				{
					21,32,
					31,42,
					41,52
				})),"Add(matrix,column).");
		}

		// Add(matrix,row)
		{
			NDArray x = NDData::New({3,2},
				{
					11,12,
					21,22,
					31,32
				});
			print(x);
			NDArray y = NDData::New({1,2},{10,20});
			print(y);
			NDArray z = x+y;
			print(z);
			Assert(z.IsEqualTo(NDData::New({3,2},
				{
					21,32,
					31,42,
					41,52
				})),"Add(matrix,row).");
		}

		// Add(matrix,matrix)
		{
			NDArray x = NDData::New({3,2},
				{
					11,21,
					31,12,
					22,32
				});
			NDArray y = NDData::New({3,2},
				{
					0.1,0.2,
					0.3,0.4,
					0.5,0.6
				});
			NDArray z = x+y;
			Assert(z.IsEqualTo(NDData::New({3,2},
				{
					11.1,21.2,
					31.3,12.4,
					22.5,32.6
				})),"Add(matrix,matrix).");
		}

		// Add(x,yT)
		{
			NDArray x = NDData::New({3,3},
				{
					1,2,3,
					4,5,6,
					7,8,9
				});
			NDArray y = NDData::New({3,3},
				{
					10,20,30,
					40,50,60,
					70,80,90
				});
			NDArray z = x+y.Transpose();
			print(z);
			Assert(z.IsEqualTo(NDData::New({3,3},
				{
					11,42,73,
					24,55,86,
					37,68,99
				})));
		}

		// Add(xT,y)
		{
			NDArray x = NDData::New({3,3},
				{
					1,2,3,
					4,5,6,
					7,8,9
				});
			NDArray y = NDData::New({3,3},
				{
					10,20,30,
					40,50,60,
					70,80,90
				});
			NDArray z = x.Transpose()+y;
			print(z);
			Assert(z.IsEqualTo(NDData::New({3,3},
				{
					11,24,37,
					42,55,68,
					73,86,99
				})));
		}
	}


	void Test_Argmax()
	{
		// ArgMax()
		{
			NDArray x = NDData::New({1,4},{1,2,4,3});
			int y = x.ArgMax();
			Assert(y==2,"ArgMax()");
		}

		// ArgMax(dim)
		{
			NDArray x = NDData::New({4,3},
				{
					91,12,13,
					21,92,23,
					31,32,93,
					94,42,43
				});
			print(x);
			NDArray y = x.ArgMax(1);	// ArgMax per row.
			print(y);
			Assert(y.IsEqualTo(NDData::New({4,1},
				{
					0,
					1,
					2,
					0
				})),"ArgMax(1): Result.");
		}
	}

	
	void Test_Asign()
	{
		{
			NDArray x = NDData::Zeros({3,4});
			print(x);
			NDArray y = NDData::New({3,4},
				{
					 1, 2, 3, 4,
					 5, 6, 7, 8,
					 9,10,11,12
				});
			x = y;
			print(y);
			Assert(x.IsEqualTo(y));
		}
		{
			NDArray x = NDData::Zeros({3,3});
			print(x);
			NDArray y = NDData::New({3,3},
				{
					1,2,3,
					4,5,6,
					7,8,9
				});
			print(y);
			x = y.Transpose();
			print(x);
			Assert(x.IsEqualTo(y.Transpose()));
		}
		{
			NDArray y = NDData::New({3,3},
				{
					1,2,3,
					4,5,6,
					7,8,9
				});
			print(y);
			NDArray x = y.Transpose();
			print(x);
			Assert(x.IsEqualTo(y.Transpose()));
			x = y;
			Assert(x.IsEqualTo(y));
		}
	}

	void Test_Cat()
	{
		{
			vector<NDArray> arrays;
			arrays.emplace_back(NDData::New({2,3,4},
				{
					1111,1112,1113,1114,
					1121,1122,1123,1124,
					1131,1132,1133,1134,

					1211,1212,1213,1214,
					1221,1222,1223,1224,
					1231,1232,1233,1234,
				}));
			arrays.emplace_back(NDData::New({2,3,4},
				{
					2111,2112,2113,2114,
					2121,2122,2123,2124,
					2131,2132,2133,2134,

					2211,2212,2213,2214,
					2221,2222,2223,2224,
					2231,2232,2233,2234,
				}));
			NDArray x = NDData::Cat(arrays,2);
			print(x);
			Assert(x.IsEqualTo(NDData::New({2,3,8},
				{
					1111,1112,1113,1114,2111,2112,2113,2114,
					1121,1122,1123,1124,2121,2122,2123,2124,
					1131,1132,1133,1134,2131,2132,2133,2134,

					1211,1212,1213,1214,2211,2212,2213,2214,
					1221,1222,1223,1224,2221,2222,2223,2224,
					1231,1232,1233,1234,2231,2232,2233,2234,
				})),"cubes_2");
		}
	}

	void Test_Div()
	{
		// matrix/column.
		{
			NDArray x = NDData::New({4,3},
				{
					  1,  4,  9,
					 16, 25, 36,
					 49, 64, 81,
					100,121,144
				});
			print(x);
			NDArray y = NDData::New({3},
				{
					  10,
					 100,
					1000
				});
			print(y);
			NDArray z = x/y;
			print(z);
			Assert(z.IsEqualTo(NDData::New({4,3},
				{
					 0.1,0.04,0.009,
					 1.6,0.25,0.036,
					 4.9,0.64,0.081,
					10.0,1.21,0.144
				})),"matrix/column.");
		}

		// cube/cube.
		{
			NDArray x = NDData::New({2,3,4},
				{
					111,112,113,114,
					121,122,123,124,

					211,212,213,214,
					221,222,223,224,

					311,312,313,314,
					321,322,323,324
				});
			print(x);

			NDArray y = NDData::New({2,3,4},
				{
					 1, 2, 3, 4,
					 5, 6, 7, 8,

					 9,10,11,12,
					13,14,15,16,

					17,18,19,20,
					21,22,23,24
				});
			print(y);

			NDArray z = x/y;
			print(z);
		
			Assert(z.IsEqualTo(NDData::New({2,3,4},
				{
					111     ,56     ,37.6667,28.5   ,
					 24.2   ,20.3333,17.5714,15.5   ,
					 23.4444,21.2   ,19.3636,17.8333,
					 17     ,15.8571,14.8667,14     ,
					 18.2941,17.3333,16.4737,15.7   ,
					 15.2857,14.6364,14.0435,13.5
				})),"cube/cube.");
		}

		// matrix/scalar.
		{
			NDArray x = NDData::New({4,3},
				{
					11,21,31,
					41,12,22,
					32,42,13,
					24,33,43
				});
			print(x);
			NDArray y = x/10;
			print(y);
			Assert(y.IsEqualTo(NDData::New({4,3},
				{
					1.1,2.1,3.1,
					4.1,1.2,2.2,
					3.2,4.2,1.3,
					2.4,3.3,4.3
				})),"matrix/scalar.");
		}

		// matrix/matrix.
		{
			NDArray x = NDData::New({4,3},
				{
					1,4,9,
					16,25,36,
					49,64,81,
					100,121,144
				});
			print(x);
			NDArray y = NDData::New({4,3},
				{
					1,2,3,
					4,5,6,
					7,8,9,
					10,11,12
				});
			print(y);
			NDArray z = x/y;
			print(z);
			Assert(z.IsEqualTo(NDData::New({4,3},
				{
					1,2,3,
					4,5,6,
					7,8,9,
					10,11,12
				})),"matrix/matrix.");
		}

		// matrix/row.
		{
			NDArray x = NDData::New({4,3},
				{
					  1,  4,  9,
					 16, 25, 36,
					 49, 64, 81,
					100,121,144
				});
			NDArray y = NDData::New({4,1},
				{
					1,10,100,1000
				});
			NDArray z = x/y;
			print(z);
			Assert(z.IsEqualTo(NDData::New({4,3},
				{
					1.0 ,4.0  ,9.0  ,
					1.6 ,2.5  ,3.6  ,
					0.49,0.64 ,0.81 ,
					0.1 ,0.121,0.144
				})),"matirx/row.");
		}
	}


	void Test_Dot()
	{
		{
			NDArray x = NDData::New({4,3},
				{
					 1, 2, 3,
					 4, 5, 6,
					 7, 8, 9,
					10,11,12
				});
			print(x,"x");
			
			NDArray y = NDData::New({3,5},
			{
				 1, 2, 3, 4, 5,
				 6, 7, 8, 9,10,
				11,12,13,14,15
			});
			print(y,"y");

			NDArray z = x.Dot(y);
			print(z,"z");

			Assert(z.IsEqualTo(NDData::New({4,5},
				{
					 46, 52, 58, 64, 70,
					100,115,130,145,160,
					154,178,202,226,250,
					208,241,274,307,340
				})));			
		}
		{
			NDArray x = NDData::New({3,4},
				{
					 1, 4, 7,10,
					 2, 5, 8,11,
					 3, 6, 9,12
				});
			NDArray xt = x.Transpose();
			print(xt,"xt");
			
			NDArray y = NDData::New({3,5},
			{
				 1, 2, 3, 4, 5,
				 6, 7, 8, 9,10,
				11,12,13,14,15
			});
			print(y,"y");

			NDArray z = xt.Dot(y);
			print(z,"z");

			Assert(z.IsEqualTo(NDData::New({4,5},
				{
					 46, 52, 58, 64, 70,
					100,115,130,145,160,
					154,178,202,226,250,
					208,241,274,307,340
				})));			
		}
		{
			NDArray x = NDData::New({4,3},
				{
					 1, 2, 3,
					 4, 5, 6,
					 7, 8, 9,
					10,11,12
				});
			print(x,"x");
			
			NDArray y = NDData::New({5,3},
			{
				1, 6,11,
				2, 7,12,
				3, 8,13,
				4, 9,14,
				5,10,15
			});
			NDArray yt = y.Transpose();
			print(yt,"yt");

			NDArray z = x.Dot(yt);
			print(z,"z");

			Assert(z.IsEqualTo(NDData::New({4,5},
				{
					 46, 52, 58, 64, 70,
					100,115,130,145,160,
					154,178,202,226,250,
					208,241,274,307,340
				})));			
		}
		{
			NDArray x = NDData::New({3,4},
				{
					 1, 4, 7,10,
					 2, 5, 8,11,
					 3, 6, 9,12
				});
			NDArray xt = x.Transpose();
			print(xt,"xt");
			
			NDArray y = NDData::New({5,3},
			{
				1, 6,11,
				2, 7,12,
				3, 8,13,
				4, 9,14,
				5,10,15
			});
			NDArray yt = y.Transpose();
			print(yt,"yt");

			NDArray z = xt.Dot(yt);
			print(z,"z");

			Assert(z.IsEqualTo(NDData::New({4,5},
				{
					 46, 52, 58, 64, 70,
					100,115,130,145,160,
					154,178,202,226,250,
					208,241,274,307,340
				})));			
		}
		{
			NDArray x = NDData::New({2,3,4},
				{
					 1, 2, 3, 4,
					 5, 6, 7, 8,
					 9,10,11,12,
				  
					13,14,15,16,
					17,18,19,20,
					21,22,23,24
				});
			print(x,"x");
			
			NDArray y = NDData::New({2,3,5},
			{
				 1, 2, 3, 4, 5,
				 6, 7, 8, 9,10,
				11,12,13,14,15,

				16,17,18,19,20,
				21,22,23,24,25,
				26,27,28,29,30
			});
			print(y,"y");

			NDArray xt = x.Transpose();
			print(xt,"xt");

			NDArray z = xt.Dot(y);
			print(z,"z");

			Assert(z.IsEqualTo(NDData::New({2,4,5},
				{
					 130,  145,  160,  175,  190,
					 148,  166,  184,  202,  220,
					 166,  187,  208,  229,  250,
					 184,  208,  232,  256,  280,

					1111, 1162, 1213, 1264, 1315,
					1174, 1228, 1282, 1336, 1390,
					1237, 1294, 1351, 1408, 1465,
					1300, 1360, 1420, 1480, 1540
				})));			
		}
		{	// slice1@x
			NDArray p = NDData::New({2,6},
				{
					 1, 2, 3, 4, 5, 6,
					 7, 8, 9,10,11,12
				});

			NDArray x = p.Slice({{},{0,3}});
			print(x);

			NDArray y = NDData::New({3,2},
				{
					1,2,
					3,4,
					5,6
				});

			NDArray z = x.Dot(y);
			print(z);

			Assert(z.IsEqualTo(NDData::New({2,2},
				{
					 22, 28,
					 76,100
				})));
		}

		{	// x@slice1
			NDArray p = NDData::New({2,6},
				{
					 1, 2, 3, 4, 5, 6,
					 7, 8, 9,10,11,12
				});

			NDArray x = NDData::New({2,2},
				{
					1,2,
					3,4
				});
			print(x,"x");

			NDArray y = p.Slice({{},{0,3}});
			print(y,"y");

			NDArray z = x.Dot(y);
			print(z,"z");

			Assert(z.IsEqualTo(NDData::New({2,3},
				{
					15,18,21,
					31,38,45
				})));
		}

		{	// x@slice2
			NDArray p = NDData::New({2,6},
				{
					 1, 2, 3, 4, 5, 6,
					 7, 8, 9,10,11,12
				});

			NDArray x = NDData::New({2,2},
				{
					1,2,
					3,4
				});

			NDArray y = p.Slice({{},{3,0}});
			print(y);

			NDArray z = x.Dot(y);
			print(z);

			Assert(z.IsEqualTo(NDData::New({2,3},
				{
					24,27,30,
					52,59,66
				})));
		}

		// Dot (2,2,3)@(3,4)=(2,2,4).
		{
			NDArray x = NDData::New({2,2,3},
				{
					111,112,113,
					121,122,123,
					211,212,213,
					221,222,223
				});
			print(x);
			NDArray y = NDData::New({3,4},
				{
					110,120,130,140,
					210,220,230,240,
					310,320,330,340
				});
			print(y);
			NDArray z = x.Dot(y);
			print(z);
			Assert(z.IsEqualTo(NDData::New({2,2,4},
				{
					 70760, 74120, 77480, 80840,
					 77060, 80720, 84380, 88040,
					133760,140120,146480,152840,
					140060,146720,153380,160040
				})),"Dot (2,2,3)@(3,4)=(2,2,4).");
		}

		// Dot (2,1,3)@(3,4)=(2,1,4).
		{
			NDArray x = NDData::New({2,1,3},
				{
					11,12,13,
					21,22,23
				});
			print(x);
			NDArray y = NDData::New({3,4},
				{
					110,120,130,140,
					210,220,230,240,
					310,320,330,340
				});
			print(y);
			NDArray z = x.Dot(y);
			print(z);
			Assert(z.IsEqualTo(NDData::New({2,1,4},
				{
					 7760, 8120, 8480, 8840,
					14060,14720,15380,16040
				})),"Dot (2,3)@(3,4)=(2,4).");
		}

		// Dot (2,3)@(3,4)=(2,4).
		{
			NDArray x = NDData::New({2,3},
				{
					11,12,13,
					21,22,23
				});
			print(x);
			NDArray y = NDData::New({3,4},
				{
					110,120,130,140,
					210,220,230,240,
					310,320,330,340
				});
			print(y);
			NDArray z = x.Dot(y);
			print(z);
			Assert(z.IsEqualTo(NDData::New({2,4},
				{
					 7760, 8120, 8480, 8840,
					14060,14720,15380,16040
				})),"Dot (2,3)@(3,4)=(2,4).");
		}

		// Dot (2,2)@(2,2)=(2,2).
		{
			NDArray x = NDData::New({2,2},
				{
					11,12,
					21,22
				});
			print(x);
			NDArray y = NDData::New({2,2},
				{
					110,120,
					210,220
				});
			print(y);
			NDArray z = x.Dot(y);
			print(z);
			Assert(z.IsEqualTo(NDData::New({2,2},
				{
					3730,3960,
					6930,7360
				})),"Dot 2x2.");
		}

		// Dot (3,3)@(3,3)=(3,3).
		{
			NDArray x = NDData::New({3,3},
				{
					11,12,13,
					21,22,23,
					31,32,33
				});
			x.Shape();
			print(x);

			NDArray y = NDData::New({3,3},
				{
					110,120,130,
					210,220,230,
					310,320,330
				});
			print(y.Shape());
			print(y);

			NDArray z = x.Dot(y);
			print(z);
			Assert(z.IsEqualTo(NDData::New({3,3},
				{
					 7760, 8120, 8480,
					14060,14720,15380,
					20360,21320,22280
				})),"Dot 3x3.");
		}

		// matrix@cube
		// (2,3)@(2,3,4)=(2,2,4)
		{
			NDArray x = NDData::New({2,3},
				{
					111,112,113,
					121,122,123
				});
			print(x.Shape());
			print(x);

			NDArray y = NDData::New({2,3,4},
				{
					111,112,113,114,
					121,122,123,124,
					131,132,133,134,

					211,212,213,214,
					221,222,223,224,
					231,232,233,234
				});
			print(y.Shape());
			print(y);

			// (2,3)@(2,3,4) = (2,2,4)
			NDArray z = x.Dot(y);
			print(z.Shape());
			print(z);
			Assert(z.IsEqualTo(NDData::New({2,2,4},
				{
					40676, 41012, 41348, 41684,
					44306, 44672, 45038, 45404,

					74276, 74612, 74948, 75284,
					80906, 81272, 81638, 82004
				})),"(2,3)@(2,3,4)=(2,2,4)");
		}

		// cube@cube
		// (1,2,3)@(2,3,4)=(2,2,4)
		{
			NDArray x = NDData::New({1,2,3},
				{
					111,112,113,
					121,122,123
				});
			print(x.Shape());
			print(x);

			NDArray y = NDData::New({2,3,4},
				{
					111,112,113,114,
					121,122,123,124,
					131,132,133,134,

					211,212,213,214,
					221,222,223,224,
					231,232,233,234
				});
			print(y.Shape());
			print(y);

			// (1,2,3)@(2,3,4) = (2,2,4)
			NDArray z = x.Dot(y);
			print(z.Shape());
			print(z);
			Assert(z.IsEqualTo(NDData::New({2,2,4},
				{
					40676, 41012, 41348, 41684,
					44306, 44672, 45038, 45404,

					74276, 74612, 74948, 75284,
					80906, 81272, 81638, 82004
				})),"(1,2,3)@(2,3,4)=(2,2,4)");
		}

		// matrix@cube
		// (5,3)@(2,3,4)=(2,5,4)
		{
			NDArray x = NDData::New({5,3},
				{
					111,112,113,
					121,122,123,
					131,132,133,
					141,142,143,
					151,152,153,
				});
			print(x.Shape());
			print(x);

			NDArray y = NDData::New({2,3,4},
				{
					111,112,113,114,
					121,122,123,124,
					131,132,133,134,

					211,212,213,214,
					221,222,223,224,
					231,232,233,234
				});
			print(y.Shape());
			print(y);

			// (5,3)@(2,3,4) = (2,5,4)
			NDArray z = x.Dot(y);
			print(z.Shape());
			print(z);
			Assert(z.IsEqualTo(NDData::New({2,5,4},
				{
					 40676, 41012, 41348, 41684,
					 44306, 44672, 45038, 45404,
					 47936, 48332, 48728, 49124,
					 51566, 51992, 52418, 52844,
					 55196, 55652, 56108, 56564,

					 74276, 74612, 74948, 75284,
					 80906, 81272, 81638, 82004,
					 87536, 87932, 88328, 88724,
					 94166, 94592, 95018, 95444,
					100796,101252,101708,102164
				})),"(5,3)@(2,3,4)=(2,2,4)");
		}

		{
			// cube@cube.
			// (2,2,3)@(2,3,4) = (2,2,4)
			NDArray x = NDData::New({2,2,3},
				{
					111,112,113,
					121,122,123,
				
					111,112,113,
					121,122,123				
				});
			print(x.Shape());
			print(x);

			NDArray y = NDData::New({2,3,4},
				{
					111,112,113,114,
					121,122,123,124,
					131,132,133,134,

					211,212,213,214,
					221,222,223,224,
					231,232,233,234
				});
			print(y.Shape());
			print(y);

			// (2,2,3)@(2,3,4) = (2,2,4)
			NDArray z = x.Dot(y);
			print(z.Shape());
			print(z);
			Assert(z.IsEqualTo(NDData::New({2,2,4},
				{
					40676, 41012, 41348, 41684,
					44306, 44672, 45038, 45404,

					74276, 74612, 74948, 75284,
					80906, 81272, 81638, 82004
				})),"(2,2,3)@(2,3,4) = (2,2,4)");
		}

		if(false)	// Lost test data file.s
		{
			NDArray x = NDData::Load("TestData\\MHA_heads(x)_0.gradient.txt");				// Inflowing gradient.
			NDArray y = NDData::Load("TestData\\MHA_head_value(x)_0.txt");					// Second parameter of dot.

			NDArray g = x.Dot(y.Transpose());

			NDArray z = NDData::Load("TestData\\MHA_head_softmax(wei)_0.gradient.txt");		// Expected gradient passed to first parameter of dot.
			Assert(g.IsEqualTo(z));
		}

		{
			NDArray x = NDData::New({3,2},
				{
					 1, 2,
					 3, 4,
					 5, 6
				});
			NDArray y = NDData::New({2,2},
				{
					 7, 8,
					 9,10
				});
			NDArray z = y.Transpose();
			NDArray w = x.Dot(z);
			print(w);
			Assert(w.IsEqualTo(NDData::New({3,2},
				{
					23,29,
					53,67,
					83,105
				})));
		}

		{
			NDArray x = NDData::New({2,2,2},
				{
					 1, 2,
					 3, 4,

					 5, 6,
					 7, 8
				});
			NDArray y = NDData::New({2,2},
				{
					 9, 10,
					 11,12
				});
			NDArray z = y.Transpose();
			NDArray w = x.Dot(z);
			print(w);
			Assert(w.IsEqualTo(NDData::New({2,2,2},
				{
					 29, 35,
					 67, 81,
					105,127,
					143,173
				})));
		}

		{
			NDArray x = NDData::New({2,2,2},
				{
					 1, 2,
					 3, 4,

					 5, 6,
					 7, 8
				});
			NDArray y = NDData::New({2,2,2},
				{
					 9, 10,
					 11,12,

					 13,14,
					 15,16
				});
			NDArray z = y.Transpose();
			NDArray w = x.Dot(z);
			print(w);
			Assert(w.IsEqualTo(NDData::New({2,2,2},
				{
					 29, 35,
					 67, 81,
					149,171,
					203,233
				})));
		}
		{
			NDArray x = NDData::New({2,2,2},
				{
					 1, 2,
					 3, 4,

					 5, 6,
					 7, 8
				});
			NDArray y = NDData::New({2,2,2},
				{
					 9, 10,
					 11,12,

					 13,14,
					 15,16
				});
			print(y);
			NDArray z1 = y.Transpose();
			print(z1);
			NDArray z2 = z1.Transpose();
			print(z2);
			NDArray w = x.Dot(z2);
			print(w);
			Assert(w.IsEqualTo(NDData::New({2,2,2},
				{
					 31, 34,
					 71, 78,
					155,166,
					211,226
				})));
		}
	}


	void Test_Gather()
	{
		{
			NDArray x = NDData::New({2,2},
				{
					-0.5,1.0,
					 0.5,2.0
				});
			print(x,"x");
			NDArray y = x.Gather(1,NDData::New({2,1},
				{
					0,
					1
				}));
			print(y,"y");
			Assert(y.IsEqualTo(NDData::New({2,1},
				{
					-0.5,
					2.0
				})));
		}

		{
			NDArray x = NDData::New({2,2},
				{
					-0.5,1.0,
					 0.5,2.0
				});
			print(x,"x");
			NDArray y = x.Gather(1,NDData::New({2,2},
				{
					0,0,
					1,0
				}));
			print(y,"y");
			Assert(y.IsEqualTo(NDData::New({2,2},
				{
					-0.5,-0.5,
					 2.0,0.5
				})));
		}
	}

	void Test_IndexSelect()
	{
		// scalar[scalar]
		{
			NDArray x = NDData::New({},{5});
			try
			{
				NDArray y = x[0];
				print(y);
				Assert(false);
			}
			catch(IncompatibleShape&){}
		}

		// vector[scalar]
		{
			NDArray x = NDData::New({5},{0,1,2,3,4});
			print(x,"x");
			NDArray y = x[2];
			print(y,"y");
			Assert(y.IsEqualTo(NDData::New({},{2})));
		}

		// matrix[scalar]
		{
			NDArray x = NDData::New({2,3},
				{
					0,1,2,
					3,4,5
				});
			print(x,"x");
			NDArray y = x[1];
			print(y,"y");
			Assert(y.IsEqualTo(NDData::New({3},{3,4,5})));
		}

		// cube[scalar]
		{
			NDArray x = NDData::New({2,3,4},
				{
					 0, 1, 2, 3,
					 4, 5, 6, 7,
					 8, 9,10,11,

					 12,13,14,15,
					 16,17,18,19,
					 20,21,22,23
				});
			print(x,"x");
			NDArray y = x[1];
			print(y,"y");
			Assert(y.IsEqualTo(NDData::New({3,4},
				{
					 12,13,14,15,
					 16,17,18,19,
					 20,21,22,23
				})));
		}

		// vector[vector].
		{
			NDArray x = NDData::New({5},{1,2,3,4,5});
			print(x,"x");
			NDArray y = NDData::New({5},{4,3,2,1,0});
			NDArray z = x[y];
			print(z,"z");
			Assert(z.IsEqualTo(NDData::New({5},{5,4,3,2,1})));
		}

		// operator[] (IndexSelect) column vector on column vector with duplicates.
		{
			NDArray x = NDData::New({5},{1,2,3,4,5});
			print(x,"x");
			NDArray y = NDData::New({5},{4,3,0,4,3});
			print(y,"y");
			NDArray z = x[y];
			print(z,"z");
			Assert(z.IsEqualTo(NDData::New({5},{5,4,1,5,4})),"operator[] (IndexSelect) column vector on column vector with duplicates.");
		}

		// operator[] (IndexSelect) column vector on row vector.
		{
			NDArray x = NDData::New({5,1},
				{
					1,
					2,
					3,
					4,
					5
				});
			print(x);
			NDArray y = NDData::New({3},{4,2,0});
			print(y);
			NDArray z = x[y];
			print(z);
			Assert(z.IsEqualTo(NDData::New({3,1},{5,3,1})),"operator[] (IndexSelect) column vector on row vector.");
		}

		// operator[] (IndexSelect) column vector on matrix.
		{
			NDArray x = NDData::New({3,3},
				{
					11,12,13,
					21,22,23,
					31,32,33
				});
			print(x);
			NDArray y = NDData::New({2},{2,0});
			NDArray z = x[y];
			NDArray w = NDData::New({2,3},
				{
					31,32,33,
					11,12,13
				});
			print(w);
			Assert(z.IsEqualTo(w),"operator[] (IndexSelect) column vector on matrix.");
		}

		// operator[] (IndexSelect) row vector on column vector.
		{
			NDArray x = NDData::New({3},{1,2,3});
			print(x);
			NDArray y = NDData::New({2,1},
				{
					2,
					0
				});
			print(y);
			NDArray z = x[y];
			print(z);
			NDArray w = NDData::New({2,1},{3,1});
			print(w);
			Assert(z.IsEqualTo(w),"operator[] (IndexSelect) row vector on column vector.");
		}

		// operator[] (IndexSelect) row vector on row vector.
		{
			NDArray x = NDData::New({3,1},
				{
					1,
					2,
					3
				});
			print(x);
			NDArray y = NDData::New({2,1},
				{
					2,
					0
				});
			print(y);
			NDArray z = x[y];
			print(z);
			NDArray w = NDData::New({2,1,1},
				{
					3,
					1
				});
			print(w);
			Assert(z.IsEqualTo(w),"operator[] (IndexSelect) row vector on row vector.");
		}

		// operator[] (IndexSelect) row vector on matrix.
		{
			NDArray x = NDData::New({3,3},
				{
					11,12,13,
					21,22,23,
					31,32,33
				});
			print(x);
			NDArray y = NDData::New({2,1},
				{
					2,
					0
				});
			print(y);
			NDArray z = x[y];
			print(z);
			NDArray w = NDData::New({2,1,3},
				{
					31,32,33,
					11,12,13
				});
			print(w);
			Assert(z.IsEqualTo(w),"operator[] (IndexSelect) row vector on matrix.");
		}
	}

	void Test_LoadWithImplicitShape()
	{
		{
			NDArray x = NDData::LoadWithImplicitShape("TestData\\Load_Scalar.txt");
			Assert(x.IsEqualTo(NDData::New({},{1.234})),"scalar");
		}

		{
			NDArray x = NDData::LoadWithImplicitShape("TestData\\Load_ColumnVector.txt");
			Assert(x.IsEqualTo(NDData::New({3},{1,2,3})));
		}

		{
			NDArray x = NDData::LoadWithImplicitShape("TestData\\Load_RowVector.txt");
			Assert(x.IsEqualTo(NDData::New({1,3},{1,2,3})));
		}

		{
			NDArray x = NDData::LoadWithImplicitShape("TestData\\Load_Matrix_2_3.txt");
			Assert(x.IsEqualTo(NDData::New({2,3},{1,2,3,4,5,6})));
		}

		{
			NDArray x = NDData::LoadWithImplicitShape("TestData\\Load_Matrix_4_3.txt");
			Assert(x.IsEqualTo(NDData::New({4,3},{1,2,3,4,5,6,7,8,9,10,11,12})));
		}

		{
			NDArray x = NDData::LoadWithImplicitShape("TestData\\Load_Cube_2_3_4.txt");
			Assert(x.IsEqualTo(NDData::New({2,3,4},
				{
					1,2,3,4,
					5,6,7,8,
					9,10,11,12,
					
					13,14,15,16,
					17,18,19,20,
					21,22,23,24
				})));
		}
	}

	void Test_Repeat()
	{
		// Repeat_Numpy(0,2) on column vector
		{
			NDArray x = NDData::New({2},{1,2});
			print(x);
			NDArray y = x.Repeat_Numpy(0,2);
			print(y);
			Assert(y.IsEqualTo(NDData::New({4},
				{
					1,1,2,2
				})),"Repeat_Numpy(0,2) on column vector");
		}

		// Repeat_Numpy(0,2) on row vector
		{
			NDArray x = NDData::New({1,2},
				{
					11,12
				});
			print(x);
			NDArray y = x.Repeat_Numpy(0,2);
			print(y);
			Assert(y.IsEqualTo(NDData::New({2,2},
				{
					11,12,
					11,12
				})),"Repeat_Numpy(0,2) on row vector");
		}

		// Repeat_Numpy(2,2) on cube.
		{
			NDArray x = NDData::New({3,2,1},
				{
					11,
					12,

					21,
					22,

					31,
					32
				});
			print(x);
			NDArray y = x.Repeat_Numpy(2,2);
			print(y);
			Assert(y.IsEqualTo(NDData::New({3,2,2},
				{
					11,11,
					12,12,

					21,21,
					22,22,

					31,31,
					32,32
				})),"Repeat_Numpy(2,2) on cube.");
		}

		// Repeat_Numpy(1,2) on single row.
		{
			NDArray x = NDData::New({1,2},
				{
					11,12
				});
			print(x);
			NDArray y = x.Repeat_Numpy(1,2);
			print(y);
			Assert(y.IsEqualTo(NDData::New({1,4},
				{
					11,11,12,12
				})),"Repeat_Numpy(1,2) on single row");
		}

		// Repeat_Numpy(1,3) on row vector.
		{
			NDArray x = NDData::New({4,1},
				{
					0.0,
					0.1,
					0.2,
					0.3
				});
			print(x);
			NDArray y = x.Repeat_Numpy(1,3);	// Make 3 copies of dim 1 (columns).
			print(y);
			Assert(y.IsEqualTo(NDData::New({4,3},
				{
					0.0,0.0,0.0,
					0.1,0.1,0.1,
					0.2,0.2,0.2,
					0.3,0.3,0.3
				})),"expand");
		}

		// Repeat_Numpy(0,2)
		{
			cout<<"NDArray::Repeat_Numpy(0,2) [2,2]."<<endl;
			NDArray x = NDData::New({2,2},
				{
					11,12,
					21,22
				});
			print(x);
			NDArray y = x.Repeat_Numpy(0,2);
			print(y);
			Assert(y.IsEqualTo(NDData::New({4,2},
				{
					11,12,
					11,12,
					21,22,
					21,22
				})),"Repeat_Numpy(0,2)");
		}

		// Repeat_Numpy(1,2)
		{
			cout<<"NDArray::Repeat_Numpy(1,2) [2,2]."<<endl;
			NDArray x = NDData::New({2,2},
				{
					11,21,
					12,22
				});
			print(x);
			NDArray y = x.Repeat_Numpy(1,2);
			print(y);
			Assert(y.IsEqualTo(NDData::New({2,4},
				{
					11,11,21,21,
					12,12,22,22
				})),"Repeat_Numpy(1,2)");
		}
	}


	void Test_Reshape()
	{
		// Reshape (B,T,C) to (B*T,C) - C vectors should be preserved.
		{
			cout<<"NDArray::Reshape((B*T,C))"<<endl;
			NDArray x = NDData::New({3,2,4},
				{
					111,112,113,114,
					121,122,123,124,
					211,212,213,214,
					221,222,223,224,
					311,312,313,314,
					321,322,323,324
				});
			cout<<"x:"<<endl;
			print(x);
			NDArray y = x.Reshape({3*2,4});
			cout<<"y:"<<endl;
			print(y);
			Assert(y[{{0},{0}}]==111,"");
			Assert(y[{{1},{1}}]==122,"");
			Assert(y[{{2},{2}}]==213,"");
			Assert(y[{{3},{3}}]==224,"");
			Assert(y[{{4},{2}}]==313,"");
			Assert(y[{{5},{1}}]==322,"");
		}

		{
			NDArray x = NDData::New({3,3},
				{
					1,2,3,
					4,5,6,
					7,8,9
				});
			print(x);
			NDArray t = x.Transpose();
			print(t);
			Assert(t.IsEqualTo(NDData::New({3,3},
				{
					1,4,7,
					2,5,8,
					3,6,9
				})));
			NDArray r1 = x.Reshape({1,9});
			print(r1);
			Assert(r1.IsEqualTo(NDData::New({1,9},{1,2,3,4,5,6,7,8,9})));
			NDArray r2 = t.Reshape({1,9});
			print(r2);
			Assert(r2.IsEqualTo(NDData::New({1,9},{1,4,7,2,5,8,3,6,9})));
			x.Slice({{0},{0}}) = NDData::New({},10);
			print(x);
			Assert(x.IsEqualTo(NDData::New({3,3},
				{
					10,2,3,
					 4,5,6,
					 7,8,9
				})));
			print(t);
			Assert(t.IsEqualTo(NDData::New({3,3},
				{
					10,4,7,
					 2,5,8,
					 3,6,9
				})));
			print(r1);
			Assert(r1.IsEqualTo(NDData::New({1,9},{10,2,3,4,5,6,7,8,9})));
			print(r2);
			Assert(r2.IsEqualTo(NDData::New({1,9},{1,4,7,2,5,8,3,6,9})));
		}
	}


	void Test_Slice()
	{
		{
			NDArray x = NDData::New({5},{0,1,2,3,4});
			print(x);
			NDArray y = x.Slice({{1,4}});
			print(y);
			Assert(y.IsEqualTo(NDData::New({3},{1,2,3})));
		}
		{
			NDArray x = NDData::New({2,3,8},
				{
					1111,1112,1113,1114,2111,2112,2113,2114,
					1121,1122,1123,1124,2121,2122,2123,2124,
					1131,1132,1133,1134,2131,2132,2133,2134,

					1211,1212,1213,1214,2211,2212,2213,2214,
					1221,1222,1223,1224,2221,2222,2223,2224,
					1231,1232,1233,1234,2231,2232,2233,2234,
				});
			NDArray y = x.Slice({{},{},{0,4}});
			print(y);
			Assert(y.IsEqualTo(NDData::New({2,3,4},
				{
					1111,1112,1113,1114,
					1121,1122,1123,1124,
					1131,1132,1133,1134,

					1211,1212,1213,1214,
					1221,1222,1223,1224,
					1231,1232,1233,1234,
				})),"cube_1");

			NDArray z = x.Slice({{},{},{4,8}});
			print(z);
			Assert(z.IsEqualTo(NDData::New({2,3,4},
				{
					2111,2112,2113,2114,
					2121,2122,2123,2124,
					2131,2132,2133,2134,

					2211,2212,2213,2214,
					2221,2222,2223,2224,
					2231,2232,2233,2234,
				})),"cube_2");
		}
		{
			// 3D with slice removing dimension 1.
			NDArray x = NDData::New({2,2,3},
				{
						 1, 2, 3,
						 4, 5, 6,
					
						 7, 8, 9,
						10,11,12
				});
			print(x);
			NDArray y = x.Slice({{},{0}});
			print(y);
			Assert(y.IsEqualTo(NDData::New({2,3},
				{
					1,2,3,
					7,8,9
				})));
		}
	}


	void Test_Softmax()
	{
	}


	void Test_Sub()
	{
		// Sub(matrix,scalar)
		{
			NDArray x = NDData::New({3,2},
				{
					11,21,
					31,12,
					22,32
				});
			print(x);
			NDArray y = x-0.5;
			print(y);
			Assert(y.IsEqualTo(NDData::New({3,2},
				{
					10.5,20.5,
					30.5,11.5,
					21.5,31.5
				})),"Sub(matrix,scalar).");
		}

		// Sub(matrix,column)
		{
			NDArray x = NDData::New({3,2},
				{
					11,12,
					21,22,
					31,32
				});
			print(x);
			NDArray y = NDData::New({2},{10,20});
			print(y);
			NDArray z = x-y;
			print(z);
			Assert(z.IsEqualTo(NDData::New({3,2},
				{
					 1,-8,
					11, 2,
					21,12
				})),"Sub(matrix,column).");
		}

		// Add(matrix,row)
		{
			NDArray x = NDData::New({3,2},
				{
					11,12,
					21,22,
					31,32
				});
			print(x);
			NDArray y = NDData::New({1,2},{10,20});
			print(y);
			NDArray z = x-y;
			print(z);
			Assert(z.IsEqualTo(NDData::New({3,2},
				{
					 1,-8,
					11, 2,
					21,12
				})),"Sub(matrix,row).");
		}

		// Add(matrix,matrix)
		{
			NDArray x = NDData::New({3,2},
				{
					11,21,
					31,12,
					22,32
				});
			print(x);
			NDArray y = NDData::New({3,2},
				{
					0.1,0.2,
					0.3,0.4,
					0.5,0.6
				});
			print(y);
			NDArray z = x-y;
			print(z);
			Assert(z.IsEqualTo(NDData::New({3,2},
				{
					10.9,20.8,
					30.7,11.6,
					21.5,31.4
				})),"Sub(matrix,matrix).");
		}
	}


	void Test_Sum()
	{
		// Sum()
		{
			NDArray x = NDData::New({2,2},{11,21,12,22});
			NDArray y = x.Sum();
			Assert(y.IsEqualTo(NDData::New({1},66.0)),"Sum()");
		}

		// Sum(0,false) on column vector.
		{
			NDArray x = NDData::New({3},{1,2,3});
			print(x);
			NDArray y = x.Sum(0,false);
			print(y);
			Assert(y.IsEqualTo(NDData::New({},{6})),"Sum(0,false) on column vector.");
		}

		// Sum(0,true) on column vector.
		{
			NDArray x = NDData::New({3},{1,2,3});
			print(x);
			NDArray y = x.Sum(0,true);
			print(y);
			Assert(y.IsEqualTo(NDData::New({1},{6})),"Sum(0,true) on column vector.");
		}

		// Sum(-1,true) on row vector.
		{
			NDArray x = NDData::New({1,3},{1,2,3});
			print(x);
			NDArray y = x.Sum(-1,true);
			print(y);
			Assert(y.IsEqualTo(NDData::New({1,1},{6})),"Sum(-1,true) on row vector.");
		}

		// Sum(-1,false) on row vector.
		{
			NDArray x = NDData::New({1,3},{1,2,3});
			print(x);
			NDArray y = x.Sum(-1,false);
			print(y);
			Assert(y.IsEqualTo(NDData::New({1},{6})),"Sum(-1,true) on row vector.");
		}

		// Sum(-1,true) on matrix.
		{
			NDArray x = NDData::New({4,3},
				{
					-0.7154, 0.6720,0.4039,
					-0.0066,-0.0338,1.0195,
					-1.1144,-1.9098,0.4209,
					 1.0   , 2.0   ,3.0
				});
			print(x);
			NDArray y = x.Sum(-1,true);
			print(y);
			Assert(y.IsEqualTo(NDData::New({4,1},
				{
					 0.3605,
			 		 0.9791,
					-2.6033,
					 6.0
				})),"Sum(-1,true) on matrix.");
		}

		// Sum(-1,false) on matrix.
		{
			NDArray x = NDData::New({4,3},
				{
					-0.7154, 0.6720,0.4039,
					-0.0066,-0.0338,1.0195,
					-1.1144,-1.9098,0.4209,
					 1.0   , 2.0   ,3.0
				});
			print(x);
			NDArray y = x.Sum(-1,false);
			print(y);
			Assert(y.IsEqualTo(NDData::New({4},
				{
					 0.3605,
			 		 0.9791,
					-2.6033,
					 6.0
				})),"Sum(-1,false) on matrix.");
		}

		// Sum(0,true) on matrix.
		{
			NDArray x = NDData::New({4,3},
				{
					-0.7154, 0.6720,0.4039,
					-0.0066,-0.0338,1.0195,
					-1.1144,-1.9098,0.4209,
					 1.0   , 2.0   ,3.0
				});
			print(x);
			NDArray y = x.Sum(0,true);
			print(y);
			Assert(y.IsEqualTo(NDData::New({1,3},
				{
					-0.8364,0.7284,4.8443
				})),"Sum(-1,true) on matrix.");
		}

		// Sum(0,false) on matrix.
		{
			NDArray x = NDData::New({4,3},
				{
					-0.7154, 0.6720,0.4039,
					-0.0066,-0.0338,1.0195,
					-1.1144,-1.9098,0.4209,
					 1.0   , 2.0   ,3.0
				});
			print(x);
			NDArray y = x.Sum(0,false);
			print(y);
			Assert(y.IsEqualTo(NDData::New({3},{-0.8364,0.7284,4.8443})),"Sum(-1,true) on matrix.");
		}

		// Sum(0,true) on cube.
		{
			NDArray x = NDData::New({2,3,4},
				{
					-0.7154, 0.6720,0.4039,0.114,
					-0.0066,-0.0338,1.0195,0.124,
					-1.1144,-1.9098,1.4209,0.134,

					-0.7154, 0.6720,0.4039,0.214,
					-0.0066,-0.0338,1.0195,0.224,
					-2.1144,-2.9098,2.4209,0.234
				});
			print(x);
			NDArray y = x.Sum(0,true);
			print(y);
			Assert(y.IsEqualTo(NDData::New({1,3,4},
				{
					-1.4308,  1.3440,  0.8078,  0.3280,
					-0.0132, -0.0676,  2.0390,  0.3480,
					-3.2288, -4.8196,  3.8418,  0.3680
				})),"Sum(0,true) on cube.");		
		}

		// Sum(0,false) on cube.
		{
			NDArray x = NDData::New({2,3,4},
				{
					-0.7154, 0.6720,0.4039,0.114,
					-0.0066,-0.0338,1.0195,0.124,
					-1.1144,-1.9098,1.4209,0.134,

					-0.7154, 0.6720,0.4039,0.214,
					-0.0066,-0.0338,1.0195,0.224,
					-2.1144,-2.9098,2.4209,0.234
				});
			print(x);
			NDArray y = x.Sum(0,false);
			print(y);
			Assert(y.IsEqualTo(NDData::New({3,4},
				{
					-1.4308,  1.3440,  0.8078,  0.3280,
					-0.0132, -0.0676,  2.0390,  0.3480,
					-3.2288, -4.8196,  3.8418,  0.3680
				})),"Sum(0,false) on cube.");		
		}

		// Sum(1,true) on cube.
		{
			NDArray x = NDData::New({2,3,4},
				{
					-0.7154, 0.6720,0.4039,0.114,
					-0.0066,-0.0338,1.0195,0.124,
					-1.1144,-1.9098,1.4209,0.134,

					-0.7154, 0.6720,0.4039,0.214,
					-0.0066,-0.0338,1.0195,0.224,
					-2.1144,-2.9098,2.4209,0.234
				});
			print(x);
			NDArray y = x.Sum(1,true);
			print(y);
			Assert(y.IsEqualTo(NDData::New({2,1,4},
				{
					-1.8364,-1.2716,2.8443,0.372,
					-2.8364,-2.2716,3.8443,0.672
				})),"Sum(1,true) on cube.");		
		}

		// Sum(1,false) on cube.
		{
			NDArray x = NDData::New({2,3,4},
				{
					-0.7154, 0.6720,0.4039,0.114,
					-0.0066,-0.0338,1.0195,0.124,
					-1.1144,-1.9098,1.4209,0.134,

					-0.7154, 0.6720,0.4039,0.214,
					-0.0066,-0.0338,1.0195,0.224,
					-2.1144,-2.9098,2.4209,0.234
				});
			print(x);
			NDArray y = x.Sum(1,false);
			print(y);
			Assert(y.IsEqualTo(NDData::New({2,4},
				{
					-1.8364,-1.2716,2.8443,0.372,
					-2.8364,-2.2716,3.8443,0.672
				})),"Sum(1,true) on cube.");		
		}

		// Sum(2,true) on cube.
		{
			NDArray x = NDData::New({2,3,4},
				{
					-0.7154, 0.6720,0.4039,0.114,
					-0.0066,-0.0338,1.0195,0.124,
					-1.1144,-1.9098,1.4209,0.134,

					-0.7154, 0.6720,0.4039,0.214,
					-0.0066,-0.0338,1.0195,0.224,
					-2.1144,-2.9098,2.4209,0.234
				});
			print(x);
			NDArray y = x.Sum(2,true);
			print(y);
			Assert(y.IsEqualTo(NDData::New({2,3,1},
				{
					0.4745,1.1031,-1.4693,
					0.5745,1.2031,-2.3693
				})),"Sum(2,true) on cube.");		
		}

		// Sum(2,true) on cube.
		{
			NDArray x = NDData::New({2,3,4},
				{
					-0.7154, 0.6720,0.4039,0.114,
					-0.0066,-0.0338,1.0195,0.124,
					-1.1144,-1.9098,1.4209,0.134,

					-0.7154, 0.6720,0.4039,0.214,
					-0.0066,-0.0338,1.0195,0.224,
					-2.1144,-2.9098,2.4209,0.234
				});
			print(x);
			NDArray y = x.Sum(2,false);
			print(y);
			Assert(y.IsEqualTo(NDData::New({2,3},
				{
					0.4745,1.1031,-1.4693,
					0.5745,1.2031,-2.3693
				})),"Sum(2,false) on cube.");		
		}
	}


	void Test_Transpose()
	{
		{
			NDArray x = NDData::New({},{1});
			print(x);
			NDArray y = x.Transpose();
			print(y);
			Assert(y.IsEqualTo(NDData::New({},{1})),"transpose()");
		}
		{
			NDArray x = NDData::New({1},{1});
			print(x);
			NDArray y = x.Transpose();
			print(y);
			Assert(y.IsEqualTo(NDData::New({1},{1})),"transpose()");
		}
		{
			NDArray x = NDData::New({1,1},
				{
					1
				});
			print(x);
			NDArray y = x.Transpose();
			print(y);
			Assert(y.IsEqualTo(NDData::New({1,1},
				{
					1
				})),"transpose(1,1)");
		}
		{
			NDArray x = NDData::New({1,2},
				{
					1,2
				});
			print(x);
			NDArray y = x.Transpose();
			print(y);
			Assert(y.IsEqualTo(NDData::New({2,1},
				{
					1,
					2
				})),"transpose(1,2)");
		}
		{
			NDArray x = NDData::New({2,1},
				{
					1,
					2
				});
			print(x);
			NDArray y = x.Transpose();
			print(y);
			Assert(y.IsEqualTo(NDData::New({1,2},
				{
					1,2
				})),"transpose(2,1)");
		}
		{
			NDArray x = NDData::New({2,2},
				{
					1,2,
					3,4
				});
			print(x);
			NDArray y = x.Transpose();
			print(y);
			Assert(y.IsEqualTo(NDData::New({2,2},
				{
					1,3,
					2,4
				})),"transpose(2,2)");
		}
		{
			NDArray x = NDData::New({2,3},
				{
					1,2,3,
					4,5,6
				});
			print(x);
			NDArray y = x.Transpose();
			print(y);
			Assert(y.IsEqualTo(NDData::New({3,2},
				{
					1,4,
					2,5,
					3,6
				})),"transpose(2,3)");
		}
		{
			NDArray x = NDData::New({3,3},
				{
					1,2,3,
					4,5,6,
					7,8,9
				});
			print(x);
			NDArray y = x.Transpose();
			print(y);
			Assert(y.IsEqualTo(NDData::New({3,3},
				{
					1,4,7,
					2,5,8,
					3,6,9
				})),"transpose(3,3)");
		}
		{
			NDArray x = NDData::New({1,3,3},
				{
					1,2,3,
					4,5,6,
					7,8,9
				});
			print(x);
			NDArray y = x.Transpose();
			print(y);
			Assert(y.IsEqualTo(NDData::New({1,3,3},
				{
					1,4,7,
					2,5,8,
					3,6,9
				})),"transpose(1,3,3)");
		}
		{
			NDArray x = NDData::New({2,3,3},
				{
					11,12,13,
					14,15,16,
					17,18,19,

					21,22,23,
					24,25,26,
					27,28,29
				});
			print(x);
			NDArray y = x.Transpose();
			print(y);
			Assert(y.IsEqualTo(NDData::New({2,3,3},
				{
					11,14,17,
					12,15,18,
					13,16,19,

					21,24,27,
					22,25,28,
					23,26,29
				})),"transpose(2,3,3)");
		}
	}
}


void Test_Unsqueeze()
{
	// Unsqueeze(-ve)
	{
		NDArray x = NDData::New({4},{1,2,3,4});
		print(x);
		NDArray y = x.Unsqueeze(-1);
		print(y);
		Assert(y.IsEqualTo(NDData::New({4,1},
			{
				1,
				2,
				3,
				4
			})),"Unsqueeze(-1).");
	}

	{
		NDArray x = NDData::New({4},{1,2,3,4});
		print(x);
		NDArray y = x.Unsqueeze(0);
		print(y);
		Assert(y.IsEqualTo(NDData::New({1,4},
			{
				1,2,3,4
			})),"Unsqueeze(0).");
	}

	{
		NDArray x = NDData::New({4},{1,2,3,4});
		print(x);
		NDArray y = x.Unsqueeze(1);
		print(y);
		Assert(y.IsEqualTo(NDData::New({4,1},
			{
				1,
				2,
				3,
				4
			})),"Unsqueeze(1).");
	}
}


void Test_NDArray()
{
	// New(shape)
	{
		NDArray x = NDData::New({3,2});
		Assert(x.Shape()==NDShape({3,2}),"New(shape): Shape.");
		Assert(x.Size()==6,"New(shape): Vector size");	// Raw memory ordering.
	}

	// New(copy)
	{
		NDArray x = NDData::New({3,2},{11,21,31,12,22,32});
		NDArray y = NDData::New(*x);
		Assert(y.Shape()==NDShape({3,2}),"New(copy): Shape.");
		Assert(x.IsEqualTo(y),"New(copy)");
	}
	
	// New(shape,vector)
	{
		NDArray x = NDData::New({3,2},{11,21,31,12,22,32});
		print(x);
		Assert(x.Shape()==NDShape({3,2}),"New(shape,vector): Shape.");
		Assert(x.IsEqualTo(NDData::New({3,2},
			{
				11,21,
				31,12,
				22,32
			})),"New(shape,vector)");	// Raw memory ordering.
	}

	// New(shape,value)
	{
		NDArray x = NDData::New({3,2},1.23);
		Assert(x.IsEqualTo(NDData::New({3,2},
			{
				1.23,1.23,
				1.23,1.23,
				1.23,1.23
			})),"New(shape,value)");
	}

	Test_Add();
	Test_Argmax();
	Test_Asign();
	Test_Cat();

	// _ClipNorm
	{
		NDArray x = NDData::New({4,3},{11,21,31,41,12,22,32,42,13,24,33,43});
		x._ClipNorm(100);
		Assert(x.IsEqualTo(NDData::New({4,3},
			{
				10.837043964422636,20.688902113897761,30.540760263372885,
				40.392618412848009,11.822229779370149,21.674087928845271,
				31.525946078320395,41.377804227795522,12.807415594317661,
				23.644459558740298,32.511131893267908,42.362990042743029
			})),"_ClipNorm[].");
	}

	Test_Div();
	Test_Dot();

	// Entropy.
	{
		NDArray x = NDData::New({2,4},
			{
				0.1,0.2,0.3,0.4,
				0.1,0.1,0.6,0.1
			});
		NDArray y = x.Entropy();
		Assert(y.Shape()==NDShape({2}),"Entropy: Shape.");
		Assert(y.IsEqualTo(NDData::New({2},{1.2798542258336676,0.99727090215780811})),"Entropy: Values.");
	}

	// Exp.
	{
		NDArray x = NDData::New({3,3},{0.11,0.12,0.13,0.21,0.22,0.23,0.31,0.32,0.33});
		NDArray y = x.Exp();
		Assert(y.IsEqualTo(NDData::New({3,3},
			{
				1.1162780704588713,1.1274968515793757,1.1388283833246218,
				1.2336780599567432,1.2460767305873808,1.2586000099294778,
				1.3634251141321778,1.3771277643359572,1.3909681284637803
			})),"Exp.");
	}

	Test_Gather();
	Test_IndexSelect();
	Test_LoadWithImplicitShape();

	// operator [] - column vector on matrix.
	{
		NDArray x = NDData::New({4,3},
			{	
				11,21,31,
				41,12,22,
				32,42,13,
				24,33,43
			});
		print(x);
		double y = x[{2,1}];
		Assert(y==42,"[vector].");
	}

	Test_Repeat();
	Test_Reshape();
	Test_Slice();
	Test_Softmax();
	Test_Sub();
	Test_Sum();
	Test_Transpose();
	Test_Unsqueeze();
}
