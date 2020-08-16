{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "_uuid": "4dbced162440c393a0a5b7e5aee344711a30e994"
   },
   "source": [
    " <h2>Facial Keypoint Detection</h2>         \n",
    " First of all let's discuss what we are given.        \n",
    "We are given three CSV files.        \n",
    "training.csv :- Its has coordinates of facial keypoints like left eye, rigth eye etc and also the image.      \n",
    "test.csv :- Its has image only and we have to give coordinates of various facial keypoints by looking at third csv file which is IdLookupTable.csv     \n",
    "Rest everything is explained below.      \n",
    "**I would really appreciate if you could upvote this kernel.**\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "_uuid": "369fa247a546e39d82bdfdc5b7d4ed58baa40e4f"
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "from IPython.display import clear_output\n",
    "from time import sleep\n",
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "_uuid": "fa1b76273d02502e3fd668dddf74ecf522044524"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['test', 'IdLookupTable.csv', 'SampleSubmission.csv', 'training']"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Train_Dir = '../input/training/training.csv'\n",
    "Test_Dir = '../input/test/test.csv'\n",
    "lookid_dir = '../input/IdLookupTable.csv'\n",
    "train_data = pd.read_csv(Train_Dir)  \n",
    "test_data = pd.read_csv(Test_Dir)\n",
    "lookid_data = pd.read_csv(lookid_dir)\n",
    "os.listdir('../input')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_uuid": "db5d8da6b196bf37a8c9934b2763ebf8dcc25667"
   },
   "source": [
    "Lets explore our dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "_uuid": "cfd045f7166f9cce2e2075b3ead83813d07012c8"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>left_eye_center_x</th>\n",
       "      <td>66.0336</td>\n",
       "      <td>64.3329</td>\n",
       "      <td>65.0571</td>\n",
       "      <td>65.2257</td>\n",
       "      <td>66.7253</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>left_eye_center_y</th>\n",
       "      <td>39.0023</td>\n",
       "      <td>34.9701</td>\n",
       "      <td>34.9096</td>\n",
       "      <td>37.2618</td>\n",
       "      <td>39.6213</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>right_eye_center_x</th>\n",
       "      <td>30.227</td>\n",
       "      <td>29.9493</td>\n",
       "      <td>30.9038</td>\n",
       "      <td>32.0231</td>\n",
       "      <td>32.2448</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>right_eye_center_y</th>\n",
       "      <td>36.4217</td>\n",
       "      <td>33.4487</td>\n",
       "      <td>34.9096</td>\n",
       "      <td>37.2618</td>\n",
       "      <td>38.042</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>left_eye_inner_corner_x</th>\n",
       "      <td>59.5821</td>\n",
       "      <td>58.8562</td>\n",
       "      <td>59.412</td>\n",
       "      <td>60.0033</td>\n",
       "      <td>58.5659</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>left_eye_inner_corner_y</th>\n",
       "      <td>39.6474</td>\n",
       "      <td>35.2743</td>\n",
       "      <td>36.321</td>\n",
       "      <td>39.1272</td>\n",
       "      <td>39.6213</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>left_eye_outer_corner_x</th>\n",
       "      <td>73.1303</td>\n",
       "      <td>70.7227</td>\n",
       "      <td>70.9844</td>\n",
       "      <td>72.3147</td>\n",
       "      <td>72.5159</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>left_eye_outer_corner_y</th>\n",
       "      <td>39.97</td>\n",
       "      <td>36.1872</td>\n",
       "      <td>36.321</td>\n",
       "      <td>38.381</td>\n",
       "      <td>39.8845</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>right_eye_inner_corner_x</th>\n",
       "      <td>36.3566</td>\n",
       "      <td>36.0347</td>\n",
       "      <td>37.6781</td>\n",
       "      <td>37.6186</td>\n",
       "      <td>36.9824</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>right_eye_inner_corner_y</th>\n",
       "      <td>37.3894</td>\n",
       "      <td>34.3615</td>\n",
       "      <td>36.321</td>\n",
       "      <td>38.7541</td>\n",
       "      <td>39.0949</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>right_eye_outer_corner_x</th>\n",
       "      <td>23.4529</td>\n",
       "      <td>24.4725</td>\n",
       "      <td>24.9764</td>\n",
       "      <td>25.3073</td>\n",
       "      <td>22.5061</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>right_eye_outer_corner_y</th>\n",
       "      <td>37.3894</td>\n",
       "      <td>33.1444</td>\n",
       "      <td>36.6032</td>\n",
       "      <td>38.0079</td>\n",
       "      <td>38.3052</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>left_eyebrow_inner_end_x</th>\n",
       "      <td>56.9533</td>\n",
       "      <td>53.9874</td>\n",
       "      <td>55.7425</td>\n",
       "      <td>56.4338</td>\n",
       "      <td>57.2496</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>left_eyebrow_inner_end_y</th>\n",
       "      <td>29.0336</td>\n",
       "      <td>28.2759</td>\n",
       "      <td>27.5709</td>\n",
       "      <td>30.9299</td>\n",
       "      <td>30.6722</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>left_eyebrow_outer_end_x</th>\n",
       "      <td>80.2271</td>\n",
       "      <td>78.6342</td>\n",
       "      <td>78.8874</td>\n",
       "      <td>77.9103</td>\n",
       "      <td>77.7629</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>left_eyebrow_outer_end_y</th>\n",
       "      <td>32.2281</td>\n",
       "      <td>30.4059</td>\n",
       "      <td>32.6516</td>\n",
       "      <td>31.6657</td>\n",
       "      <td>31.7372</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>right_eyebrow_inner_end_x</th>\n",
       "      <td>40.2276</td>\n",
       "      <td>42.7289</td>\n",
       "      <td>42.1939</td>\n",
       "      <td>41.6715</td>\n",
       "      <td>38.0354</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>right_eyebrow_inner_end_y</th>\n",
       "      <td>29.0023</td>\n",
       "      <td>26.146</td>\n",
       "      <td>28.1355</td>\n",
       "      <td>31.05</td>\n",
       "      <td>30.9354</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>right_eyebrow_outer_end_x</th>\n",
       "      <td>16.3564</td>\n",
       "      <td>16.8654</td>\n",
       "      <td>16.7912</td>\n",
       "      <td>20.458</td>\n",
       "      <td>15.9259</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>right_eyebrow_outer_end_y</th>\n",
       "      <td>29.6475</td>\n",
       "      <td>27.0589</td>\n",
       "      <td>32.0871</td>\n",
       "      <td>29.9093</td>\n",
       "      <td>30.6722</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>nose_tip_x</th>\n",
       "      <td>44.4206</td>\n",
       "      <td>48.2063</td>\n",
       "      <td>47.5573</td>\n",
       "      <td>51.8851</td>\n",
       "      <td>43.2995</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>nose_tip_y</th>\n",
       "      <td>57.0668</td>\n",
       "      <td>55.6609</td>\n",
       "      <td>53.5389</td>\n",
       "      <td>54.1665</td>\n",
       "      <td>64.8895</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mouth_left_corner_x</th>\n",
       "      <td>61.1953</td>\n",
       "      <td>56.4214</td>\n",
       "      <td>60.8229</td>\n",
       "      <td>65.5989</td>\n",
       "      <td>60.6714</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mouth_left_corner_y</th>\n",
       "      <td>79.9702</td>\n",
       "      <td>76.352</td>\n",
       "      <td>73.0143</td>\n",
       "      <td>72.7037</td>\n",
       "      <td>77.5232</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mouth_right_corner_x</th>\n",
       "      <td>28.6145</td>\n",
       "      <td>35.1224</td>\n",
       "      <td>33.7263</td>\n",
       "      <td>37.2455</td>\n",
       "      <td>31.1918</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mouth_right_corner_y</th>\n",
       "      <td>77.389</td>\n",
       "      <td>76.0477</td>\n",
       "      <td>72.732</td>\n",
       "      <td>74.1955</td>\n",
       "      <td>76.9973</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mouth_center_top_lip_x</th>\n",
       "      <td>43.3126</td>\n",
       "      <td>46.6846</td>\n",
       "      <td>47.2749</td>\n",
       "      <td>50.3032</td>\n",
       "      <td>44.9627</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mouth_center_top_lip_y</th>\n",
       "      <td>72.9355</td>\n",
       "      <td>70.2666</td>\n",
       "      <td>70.1918</td>\n",
       "      <td>70.0917</td>\n",
       "      <td>73.7074</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mouth_center_bottom_lip_x</th>\n",
       "      <td>43.1307</td>\n",
       "      <td>45.4679</td>\n",
       "      <td>47.2749</td>\n",
       "      <td>51.5612</td>\n",
       "      <td>44.2271</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mouth_center_bottom_lip_y</th>\n",
       "      <td>84.4858</td>\n",
       "      <td>85.4802</td>\n",
       "      <td>78.6594</td>\n",
       "      <td>78.2684</td>\n",
       "      <td>86.8712</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Image</th>\n",
       "      <td>238 236 237 238 240 240 239 241 241 243 240 23...</td>\n",
       "      <td>219 215 204 196 204 211 212 200 180 168 178 19...</td>\n",
       "      <td>144 142 159 180 188 188 184 180 167 132 84 59 ...</td>\n",
       "      <td>193 192 193 194 194 194 193 192 168 111 50 12 ...</td>\n",
       "      <td>147 148 160 196 215 214 216 217 219 220 206 18...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                                           0                        ...                                                                          4\n",
       "left_eye_center_x                                                    66.0336                        ...                                                                    66.7253\n",
       "left_eye_center_y                                                    39.0023                        ...                                                                    39.6213\n",
       "right_eye_center_x                                                    30.227                        ...                                                                    32.2448\n",
       "right_eye_center_y                                                   36.4217                        ...                                                                     38.042\n",
       "left_eye_inner_corner_x                                              59.5821                        ...                                                                    58.5659\n",
       "left_eye_inner_corner_y                                              39.6474                        ...                                                                    39.6213\n",
       "left_eye_outer_corner_x                                              73.1303                        ...                                                                    72.5159\n",
       "left_eye_outer_corner_y                                                39.97                        ...                                                                    39.8845\n",
       "right_eye_inner_corner_x                                             36.3566                        ...                                                                    36.9824\n",
       "right_eye_inner_corner_y                                             37.3894                        ...                                                                    39.0949\n",
       "right_eye_outer_corner_x                                             23.4529                        ...                                                                    22.5061\n",
       "right_eye_outer_corner_y                                             37.3894                        ...                                                                    38.3052\n",
       "left_eyebrow_inner_end_x                                             56.9533                        ...                                                                    57.2496\n",
       "left_eyebrow_inner_end_y                                             29.0336                        ...                                                                    30.6722\n",
       "left_eyebrow_outer_end_x                                             80.2271                        ...                                                                    77.7629\n",
       "left_eyebrow_outer_end_y                                             32.2281                        ...                                                                    31.7372\n",
       "right_eyebrow_inner_end_x                                            40.2276                        ...                                                                    38.0354\n",
       "right_eyebrow_inner_end_y                                            29.0023                        ...                                                                    30.9354\n",
       "right_eyebrow_outer_end_x                                            16.3564                        ...                                                                    15.9259\n",
       "right_eyebrow_outer_end_y                                            29.6475                        ...                                                                    30.6722\n",
       "nose_tip_x                                                           44.4206                        ...                                                                    43.2995\n",
       "nose_tip_y                                                           57.0668                        ...                                                                    64.8895\n",
       "mouth_left_corner_x                                                  61.1953                        ...                                                                    60.6714\n",
       "mouth_left_corner_y                                                  79.9702                        ...                                                                    77.5232\n",
       "mouth_right_corner_x                                                 28.6145                        ...                                                                    31.1918\n",
       "mouth_right_corner_y                                                  77.389                        ...                                                                    76.9973\n",
       "mouth_center_top_lip_x                                               43.3126                        ...                                                                    44.9627\n",
       "mouth_center_top_lip_y                                               72.9355                        ...                                                                    73.7074\n",
       "mouth_center_bottom_lip_x                                            43.1307                        ...                                                                    44.2271\n",
       "mouth_center_bottom_lip_y                                            84.4858                        ...                                                                    86.8712\n",
       "Image                      238 236 237 238 240 240 239 241 241 243 240 23...                        ...                          147 148 160 196 215 214 216 217 219 220 206 18...\n",
       "\n",
       "[31 rows x 5 columns]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data.head().T"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_uuid": "14ae8eb84c1ed40db949d6dddeba331b2ed84487"
   },
   "source": [
    "Lets check for missing values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "_uuid": "67368f645afe618d6fc717552a6847de7e5ec66a"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "True     28\n",
       "False     3\n",
       "dtype: int64"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data.isnull().any().value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_uuid": "f55b89f149e6bbc79d0e9c636c64c5c6fc7192b2"
   },
   "source": [
    "So there are missing values in 28 columns. We can do two things here one remove the rows having missing values and another is the fill missing values with something. I used two option as removing rows will reduce our dataset. \n",
    "I filled the missing values with the previous values in that row."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "_uuid": "69165acb462a9b47f22fe81a8fe8eaaca4f518d2"
   },
   "outputs": [],
   "source": [
    "\n",
    "train_data.fillna(method = 'ffill',inplace = True)\n",
    "#train_data.reset_index(drop = True,inplace = True)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_uuid": "e6cd1f4c243b44ecc0530bf416d761930a4f94d1"
   },
   "source": [
    "Lets check for missing values now"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "_uuid": "29ca4e12ec805d837ec4eac91cb91d5a133f8740"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "False    31\n",
       "dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data.isnull().any().value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_uuid": "e1b88f1528838c0a8fec61f9a02a70b5077312e9"
   },
   "source": [
    "As there is no missing values we can now separate the labels and features.\n",
    "The image is our feature and other values are labes that we have to predict later.\n",
    "As image column values are in string format and there is also some missing values so we have to split the string by space and append it and also handling missing values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "_uuid": "e78ca4523425835f1b584f3e30e5c9dcc8014253"
   },
   "outputs": [],
   "source": [
    "\n",
    "imag = []\n",
    "for i in range(0,7049):\n",
    "    img = train_data['Image'][i].split(' ')\n",
    "    img = ['0' if x == '' else x for x in img]\n",
    "    imag.append(img)\n",
    "    \n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_uuid": "e112a93d30f86687f80fb01d9f916633d87682db"
   },
   "source": [
    "Lets reshape and convert it into float value."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "_uuid": "da09436050dc2df5da7cb49ad90125b1b756f1a4"
   },
   "outputs": [],
   "source": [
    "image_list = np.array(imag,dtype = 'float')\n",
    "X_train = image_list.reshape(-1,96,96,1)\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_uuid": "58a352bb3b38dadf37da1a6b62798fe1e3df8614",
    "collapsed": true
   },
   "source": [
    "Lets see what is the first image."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "_uuid": "3e953b9fa753d6f7c1092a2d8d2faf6cff5a4f93"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.imshow(X_train[0].reshape(96,96),cmap='gray')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_uuid": "a833f4cc5e559774d3a310fd09d40d31e49e71da"
   },
   "source": [
    "Now lets separate labels."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "_uuid": "e9d804a035809cdf8ffda19f41ce3feb278a38fb"
   },
   "outputs": [],
   "source": [
    "training = train_data.drop('Image',axis = 1)\n",
    "\n",
    "y_train = []\n",
    "for i in range(0,7049):\n",
    "    y = training.iloc[i,:]\n",
    "\n",
    "    y_train.append(y)\n",
    "y_train = np.array(y_train,dtype = 'float')\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_uuid": "d1ade36c2a9ffc30411987384691c49a8859818b"
   },
   "source": [
    "As our data is ready for training , lets define our model. I am using keras and simple dense layers. For loss function I am using 'mse' ( mean squared error ) as we have to predict new values. Our result evaluted on the basics of 'mae' ( mean absolute error ) . "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "_uuid": "182dbbf5e211249cd5087f2e1218c7000c15e8d7"
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "from keras.layers import Conv2D,Dropout,Dense,Flatten\n",
    "from keras.models import Sequential\n",
    "\n",
    "model = Sequential([Flatten(input_shape=(96,96)),\n",
    "                         Dense(128, activation=\"relu\"),\n",
    "                         Dropout(0.1),\n",
    "                         Dense(64, activation=\"relu\"),\n",
    "                         Dense(30)\n",
    "                         ])\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.layers.advanced_activations import LeakyReLU\n",
    "from keras.models import Sequential, Model\n",
    "from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "conv2d_1 (Conv2D)            (None, 96, 96, 32)        288       \n",
      "_________________________________________________________________\n",
      "leaky_re_lu_1 (LeakyReLU)    (None, 96, 96, 32)        0         \n",
      "_________________________________________________________________\n",
      "batch_normalization_1 (Batch (None, 96, 96, 32)        128       \n",
      "_________________________________________________________________\n",
      "conv2d_2 (Conv2D)            (None, 96, 96, 32)        9216      \n",
      "_________________________________________________________________\n",
      "leaky_re_lu_2 (LeakyReLU)    (None, 96, 96, 32)        0         \n",
      "_________________________________________________________________\n",
      "batch_normalization_2 (Batch (None, 96, 96, 32)        128       \n",
      "_________________________________________________________________\n",
      "max_pooling2d_1 (MaxPooling2 (None, 48, 48, 32)        0         \n",
      "_________________________________________________________________\n",
      "conv2d_3 (Conv2D)            (None, 48, 48, 64)        18432     \n",
      "_________________________________________________________________\n",
      "leaky_re_lu_3 (LeakyReLU)    (None, 48, 48, 64)        0         \n",
      "_________________________________________________________________\n",
      "batch_normalization_3 (Batch (None, 48, 48, 64)        256       \n",
      "_________________________________________________________________\n",
      "conv2d_4 (Conv2D)            (None, 48, 48, 64)        36864     \n",
      "_________________________________________________________________\n",
      "leaky_re_lu_4 (LeakyReLU)    (None, 48, 48, 64)        0         \n",
      "_________________________________________________________________\n",
      "batch_normalization_4 (Batch (None, 48, 48, 64)        256       \n",
      "_________________________________________________________________\n",
      "max_pooling2d_2 (MaxPooling2 (None, 24, 24, 64)        0         \n",
      "_________________________________________________________________\n",
      "conv2d_5 (Conv2D)            (None, 24, 24, 96)        55296     \n",
      "_________________________________________________________________\n",
      "leaky_re_lu_5 (LeakyReLU)    (None, 24, 24, 96)        0         \n",
      "_________________________________________________________________\n",
      "batch_normalization_5 (Batch (None, 24, 24, 96)        384       \n",
      "_________________________________________________________________\n",
      "conv2d_6 (Conv2D)            (None, 24, 24, 96)        82944     \n",
      "_________________________________________________________________\n",
      "leaky_re_lu_6 (LeakyReLU)    (None, 24, 24, 96)        0         \n",
      "_________________________________________________________________\n",
      "batch_normalization_6 (Batch (None, 24, 24, 96)        384       \n",
      "_________________________________________________________________\n",
      "max_pooling2d_3 (MaxPooling2 (None, 12, 12, 96)        0         \n",
      "_________________________________________________________________\n",
      "conv2d_7 (Conv2D)            (None, 12, 12, 128)       110592    \n",
      "_________________________________________________________________\n",
      "leaky_re_lu_7 (LeakyReLU)    (None, 12, 12, 128)       0         \n",
      "_________________________________________________________________\n",
      "batch_normalization_7 (Batch (None, 12, 12, 128)       512       \n",
      "_________________________________________________________________\n",
      "conv2d_8 (Conv2D)            (None, 12, 12, 128)       147456    \n",
      "_________________________________________________________________\n",
      "leaky_re_lu_8 (LeakyReLU)    (None, 12, 12, 128)       0         \n",
      "_________________________________________________________________\n",
      "batch_normalization_8 (Batch (None, 12, 12, 128)       512       \n",
      "_________________________________________________________________\n",
      "max_pooling2d_4 (MaxPooling2 (None, 6, 6, 128)         0         \n",
      "_________________________________________________________________\n",
      "conv2d_9 (Conv2D)            (None, 6, 6, 256)         294912    \n",
      "_________________________________________________________________\n",
      "leaky_re_lu_9 (LeakyReLU)    (None, 6, 6, 256)         0         \n",
      "_________________________________________________________________\n",
      "batch_normalization_9 (Batch (None, 6, 6, 256)         1024      \n",
      "_________________________________________________________________\n",
      "conv2d_10 (Conv2D)           (None, 6, 6, 256)         589824    \n",
      "_________________________________________________________________\n",
      "leaky_re_lu_10 (LeakyReLU)   (None, 6, 6, 256)         0         \n",
      "_________________________________________________________________\n",
      "batch_normalization_10 (Batc (None, 6, 6, 256)         1024      \n",
      "_________________________________________________________________\n",
      "max_pooling2d_5 (MaxPooling2 (None, 3, 3, 256)         0         \n",
      "_________________________________________________________________\n",
      "conv2d_11 (Conv2D)           (None, 3, 3, 512)         1179648   \n",
      "_________________________________________________________________\n",
      "leaky_re_lu_11 (LeakyReLU)   (None, 3, 3, 512)         0         \n",
      "_________________________________________________________________\n",
      "batch_normalization_11 (Batc (None, 3, 3, 512)         2048      \n",
      "_________________________________________________________________\n",
      "conv2d_12 (Conv2D)           (None, 3, 3, 512)         2359296   \n",
      "_________________________________________________________________\n",
      "leaky_re_lu_12 (LeakyReLU)   (None, 3, 3, 512)         0         \n",
      "_________________________________________________________________\n",
      "batch_normalization_12 (Batc (None, 3, 3, 512)         2048      \n",
      "_________________________________________________________________\n",
      "flatten_2 (Flatten)          (None, 4608)              0         \n",
      "_________________________________________________________________\n",
      "dense_4 (Dense)              (None, 512)               2359808   \n",
      "_________________________________________________________________\n",
      "dropout_2 (Dropout)          (None, 512)               0         \n",
      "_________________________________________________________________\n",
      "dense_5 (Dense)              (None, 30)                15390     \n",
      "=================================================================\n",
      "Total params: 7,268,670\n",
      "Trainable params: 7,264,318\n",
      "Non-trainable params: 4,352\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model = Sequential()\n",
    "\n",
    "model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(96,96,1)))\n",
    "model.add(LeakyReLU(alpha = 0.1))\n",
    "model.add(BatchNormalization())\n",
    "\n",
    "model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))\n",
    "model.add(LeakyReLU(alpha = 0.1))\n",
    "model.add(BatchNormalization())\n",
    "model.add(MaxPool2D(pool_size=(2, 2)))\n",
    "\n",
    "model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))\n",
    "model.add(LeakyReLU(alpha = 0.1))\n",
    "model.add(BatchNormalization())\n",
    "\n",
    "model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))\n",
    "model.add(LeakyReLU(alpha = 0.1))\n",
    "model.add(BatchNormalization())\n",
    "model.add(MaxPool2D(pool_size=(2, 2)))\n",
    "\n",
    "model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))\n",
    "model.add(LeakyReLU(alpha = 0.1))\n",
    "model.add(BatchNormalization())\n",
    "\n",
    "model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))\n",
    "model.add(LeakyReLU(alpha = 0.1))\n",
    "model.add(BatchNormalization())\n",
    "model.add(MaxPool2D(pool_size=(2, 2)))\n",
    "\n",
    "model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))\n",
    "# model.add(BatchNormalization())\n",
    "model.add(LeakyReLU(alpha = 0.1))\n",
    "model.add(BatchNormalization())\n",
    "\n",
    "model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))\n",
    "model.add(LeakyReLU(alpha = 0.1))\n",
    "model.add(BatchNormalization())\n",
    "model.add(MaxPool2D(pool_size=(2, 2)))\n",
    "\n",
    "model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))\n",
    "model.add(LeakyReLU(alpha = 0.1))\n",
    "model.add(BatchNormalization())\n",
    "\n",
    "model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))\n",
    "model.add(LeakyReLU(alpha = 0.1))\n",
    "model.add(BatchNormalization())\n",
    "model.add(MaxPool2D(pool_size=(2, 2)))\n",
    "\n",
    "model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))\n",
    "model.add(LeakyReLU(alpha = 0.1))\n",
    "model.add(BatchNormalization())\n",
    "\n",
    "model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))\n",
    "model.add(LeakyReLU(alpha = 0.1))\n",
    "model.add(BatchNormalization())\n",
    "\n",
    "\n",
    "model.add(Flatten())\n",
    "model.add(Dense(512,activation='relu'))\n",
    "model.add(Dropout(0.1))\n",
    "model.add(Dense(30))\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer='adam', \n",
    "              loss='mean_squared_error',\n",
    "              metrics=['mae'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_uuid": "e4cf4686b410841f2e34dbb081f3429d1b0f67e9"
   },
   "source": [
    "Now our model is defined and we will train it by calling fit method. I ran it for 500 iteration keeping batch size and validtion set size as 20% ( 20% of the training data will be kept for validating the model )."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "_uuid": "894af9cbfcf2dca50e7407946cad318157b77d0a"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train on 5639 samples, validate on 1410 samples\n",
      "Epoch 1/50\n",
      "5639/5639 [==============================] - 10s 2ms/step - loss: 377.6182 - mean_absolute_error: 12.6949 - val_loss: 1379.9186 - val_mean_absolute_error: 34.0214\n",
      "Epoch 2/50\n",
      "5639/5639 [==============================] - 4s 714us/step - loss: 33.4846 - mean_absolute_error: 4.5120 - val_loss: 155.8508 - val_mean_absolute_error: 11.2401\n",
      "Epoch 3/50\n",
      "5639/5639 [==============================] - 4s 714us/step - loss: 27.0987 - mean_absolute_error: 4.0227 - val_loss: 43.9558 - val_mean_absolute_error: 5.6561\n",
      "Epoch 4/50\n",
      "5639/5639 [==============================] - 4s 713us/step - loss: 24.3302 - mean_absolute_error: 3.7967 - val_loss: 14.7447 - val_mean_absolute_error: 3.1370\n",
      "Epoch 5/50\n",
      "5639/5639 [==============================] - 4s 713us/step - loss: 22.8195 - mean_absolute_error: 3.6605 - val_loss: 11.0507 - val_mean_absolute_error: 2.5203\n",
      "Epoch 6/50\n",
      "5639/5639 [==============================] - 4s 713us/step - loss: 21.2416 - mean_absolute_error: 3.5235 - val_loss: 5.9666 - val_mean_absolute_error: 1.5803\n",
      "Epoch 7/50\n",
      "5639/5639 [==============================] - 4s 714us/step - loss: 19.5302 - mean_absolute_error: 3.3716 - val_loss: 5.7150 - val_mean_absolute_error: 1.4963\n",
      "Epoch 8/50\n",
      "5639/5639 [==============================] - 4s 714us/step - loss: 18.1436 - mean_absolute_error: 3.2441 - val_loss: 16.1095 - val_mean_absolute_error: 3.2201\n",
      "Epoch 9/50\n",
      "5639/5639 [==============================] - 4s 713us/step - loss: 16.7931 - mean_absolute_error: 3.1157 - val_loss: 10.9232 - val_mean_absolute_error: 2.3812\n",
      "Epoch 10/50\n",
      "5639/5639 [==============================] - 4s 713us/step - loss: 15.7416 - mean_absolute_error: 3.0174 - val_loss: 10.5420 - val_mean_absolute_error: 2.2542\n",
      "Epoch 11/50\n",
      "5639/5639 [==============================] - 4s 712us/step - loss: 15.0570 - mean_absolute_error: 2.9476 - val_loss: 6.0849 - val_mean_absolute_error: 1.6582\n",
      "Epoch 12/50\n",
      "5639/5639 [==============================] - 4s 712us/step - loss: 15.4552 - mean_absolute_error: 2.9932 - val_loss: 6.8587 - val_mean_absolute_error: 1.6980\n",
      "Epoch 13/50\n",
      "5639/5639 [==============================] - 4s 712us/step - loss: 13.5468 - mean_absolute_error: 2.7893 - val_loss: 8.6275 - val_mean_absolute_error: 1.9408\n",
      "Epoch 14/50\n",
      "5639/5639 [==============================] - 4s 714us/step - loss: 13.4385 - mean_absolute_error: 2.7744 - val_loss: 18.7911 - val_mean_absolute_error: 3.3173\n",
      "Epoch 15/50\n",
      "5639/5639 [==============================] - 4s 713us/step - loss: 14.3192 - mean_absolute_error: 2.8772 - val_loss: 10.1248 - val_mean_absolute_error: 2.2931\n",
      "Epoch 16/50\n",
      "5639/5639 [==============================] - 4s 713us/step - loss: 12.7763 - mean_absolute_error: 2.7111 - val_loss: 6.4457 - val_mean_absolute_error: 1.7514\n",
      "Epoch 17/50\n",
      "5639/5639 [==============================] - 4s 714us/step - loss: 12.2629 - mean_absolute_error: 2.6564 - val_loss: 7.4609 - val_mean_absolute_error: 2.1679\n",
      "Epoch 18/50\n",
      "5639/5639 [==============================] - 4s 713us/step - loss: 11.8845 - mean_absolute_error: 2.6119 - val_loss: 11.6972 - val_mean_absolute_error: 2.6633\n",
      "Epoch 19/50\n",
      "5639/5639 [==============================] - 4s 712us/step - loss: 13.1595 - mean_absolute_error: 2.7611 - val_loss: 12.7654 - val_mean_absolute_error: 2.5166\n",
      "Epoch 20/50\n",
      "5639/5639 [==============================] - 4s 713us/step - loss: 11.8998 - mean_absolute_error: 2.6057 - val_loss: 16.6643 - val_mean_absolute_error: 3.4611\n",
      "Epoch 21/50\n",
      "5639/5639 [==============================] - 4s 714us/step - loss: 12.8750 - mean_absolute_error: 2.7263 - val_loss: 4.6833 - val_mean_absolute_error: 1.5349\n",
      "Epoch 22/50\n",
      "5639/5639 [==============================] - 4s 714us/step - loss: 10.3961 - mean_absolute_error: 2.4402 - val_loss: 4.7823 - val_mean_absolute_error: 1.4450\n",
      "Epoch 23/50\n",
      "5639/5639 [==============================] - 4s 714us/step - loss: 9.8129 - mean_absolute_error: 2.3665 - val_loss: 3.5518 - val_mean_absolute_error: 1.1196\n",
      "Epoch 24/50\n",
      "5639/5639 [==============================] - 4s 715us/step - loss: 9.8545 - mean_absolute_error: 2.3669 - val_loss: 12.2720 - val_mean_absolute_error: 2.6883\n",
      "Epoch 25/50\n",
      "5639/5639 [==============================] - 4s 713us/step - loss: 10.2052 - mean_absolute_error: 2.4207 - val_loss: 4.1593 - val_mean_absolute_error: 1.2567\n",
      "Epoch 26/50\n",
      "5639/5639 [==============================] - 4s 713us/step - loss: 9.1432 - mean_absolute_error: 2.2808 - val_loss: 6.4687 - val_mean_absolute_error: 2.0475\n",
      "Epoch 27/50\n",
      "5639/5639 [==============================] - 4s 713us/step - loss: 10.1386 - mean_absolute_error: 2.4232 - val_loss: 19.4606 - val_mean_absolute_error: 3.6711\n",
      "Epoch 28/50\n",
      "5639/5639 [==============================] - 4s 714us/step - loss: 11.7821 - mean_absolute_error: 2.6093 - val_loss: 13.4893 - val_mean_absolute_error: 3.1026\n",
      "Epoch 29/50\n",
      "5639/5639 [==============================] - 4s 714us/step - loss: 10.5918 - mean_absolute_error: 2.4735 - val_loss: 4.0913 - val_mean_absolute_error: 1.4125\n",
      "Epoch 30/50\n",
      "5639/5639 [==============================] - 4s 715us/step - loss: 8.6489 - mean_absolute_error: 2.2213 - val_loss: 5.6860 - val_mean_absolute_error: 1.6940\n",
      "Epoch 31/50\n",
      "5639/5639 [==============================] - 4s 713us/step - loss: 8.2803 - mean_absolute_error: 2.1739 - val_loss: 4.6593 - val_mean_absolute_error: 1.3612\n",
      "Epoch 32/50\n",
      "5639/5639 [==============================] - 4s 714us/step - loss: 8.9221 - mean_absolute_error: 2.2677 - val_loss: 17.0307 - val_mean_absolute_error: 3.4863\n",
      "Epoch 33/50\n",
      "5639/5639 [==============================] - 4s 714us/step - loss: 12.1591 - mean_absolute_error: 2.6706 - val_loss: 6.3484 - val_mean_absolute_error: 1.9706\n",
      "Epoch 34/50\n",
      "5639/5639 [==============================] - 4s 715us/step - loss: 8.0727 - mean_absolute_error: 2.1537 - val_loss: 12.2127 - val_mean_absolute_error: 2.7651\n",
      "Epoch 35/50\n",
      "5639/5639 [==============================] - 4s 714us/step - loss: 9.6022 - mean_absolute_error: 2.3528 - val_loss: 4.6066 - val_mean_absolute_error: 1.3896\n",
      "Epoch 36/50\n",
      "5639/5639 [==============================] - 4s 714us/step - loss: 7.4830 - mean_absolute_error: 2.0619 - val_loss: 5.5509 - val_mean_absolute_error: 1.7261\n",
      "Epoch 37/50\n",
      "5639/5639 [==============================] - 4s 713us/step - loss: 7.5862 - mean_absolute_error: 2.0898 - val_loss: 13.0215 - val_mean_absolute_error: 2.9523\n",
      "Epoch 38/50\n",
      "5639/5639 [==============================] - 4s 715us/step - loss: 9.2946 - mean_absolute_error: 2.3288 - val_loss: 3.2262 - val_mean_absolute_error: 1.1084\n",
      "Epoch 39/50\n",
      "5639/5639 [==============================] - 4s 716us/step - loss: 6.8063 - mean_absolute_error: 1.9717 - val_loss: 4.4088 - val_mean_absolute_error: 1.3327\n",
      "Epoch 40/50\n",
      "5639/5639 [==============================] - 4s 716us/step - loss: 6.9705 - mean_absolute_error: 2.0020 - val_loss: 3.4662 - val_mean_absolute_error: 1.2352\n",
      "Epoch 41/50\n",
      "5639/5639 [==============================] - 4s 714us/step - loss: 6.8596 - mean_absolute_error: 1.9859 - val_loss: 9.3216 - val_mean_absolute_error: 2.4667\n",
      "Epoch 42/50\n",
      "5639/5639 [==============================] - 4s 714us/step - loss: 7.4958 - mean_absolute_error: 2.0792 - val_loss: 16.8052 - val_mean_absolute_error: 3.3981\n",
      "Epoch 43/50\n",
      "5639/5639 [==============================] - 4s 714us/step - loss: 10.4684 - mean_absolute_error: 2.4816 - val_loss: 8.1681 - val_mean_absolute_error: 2.3264\n",
      "Epoch 44/50\n",
      "5639/5639 [==============================] - 4s 713us/step - loss: 8.0972 - mean_absolute_error: 2.1603 - val_loss: 5.7202 - val_mean_absolute_error: 1.7263\n",
      "Epoch 45/50\n",
      "5639/5639 [==============================] - 4s 713us/step - loss: 7.4105 - mean_absolute_error: 2.0711 - val_loss: 2.6523 - val_mean_absolute_error: 0.9367\n",
      "Epoch 46/50\n",
      "5639/5639 [==============================] - 4s 714us/step - loss: 6.6229 - mean_absolute_error: 1.9573 - val_loss: 3.4929 - val_mean_absolute_error: 1.2566\n",
      "Epoch 47/50\n",
      "5639/5639 [==============================] - 4s 714us/step - loss: 5.9132 - mean_absolute_error: 1.8473 - val_loss: 3.6761 - val_mean_absolute_error: 1.3301\n",
      "Epoch 48/50\n",
      "5639/5639 [==============================] - 4s 715us/step - loss: 5.6835 - mean_absolute_error: 1.8076 - val_loss: 10.4315 - val_mean_absolute_error: 2.8098\n",
      "Epoch 49/50\n",
      "5639/5639 [==============================] - 4s 714us/step - loss: 9.1544 - mean_absolute_error: 2.3395 - val_loss: 5.4645 - val_mean_absolute_error: 1.8136\n",
      "Epoch 50/50\n",
      "5639/5639 [==============================] - 4s 713us/step - loss: 7.2843 - mean_absolute_error: 2.0642 - val_loss: 4.6411 - val_mean_absolute_error: 1.1547\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7feea00acc88>"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X_train,y_train,epochs = 50,batch_size = 256,validation_split = 0.2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_uuid": "cf7e21d1b2e1282636b8bca23d1297ec43642179"
   },
   "source": [
    "Now lets prepare our testing data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "_uuid": "587e6f2a158cccc7b8de4df99885c9878c6a5683"
   },
   "outputs": [],
   "source": [
    "#preparing test data\n",
    "timag = []\n",
    "for i in range(0,1783):\n",
    "    timg = test_data['Image'][i].split(' ')\n",
    "    timg = ['0' if x == '' else x for x in timg]\n",
    "    \n",
    "    timag.append(timg)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_uuid": "88db8ebf9a5e18eb120ed1808ec361ff53bdc7be"
   },
   "source": [
    "Reshaping and converting "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "_uuid": "1203ed3b00d70e52de0ac457facfb774a11f5816"
   },
   "outputs": [],
   "source": [
    "timage_list = np.array(timag,dtype = 'float')\n",
    "X_test = timage_list.reshape(-1,96,96,1) "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_uuid": "d6733e0e2d3fa1384f84bdfaa143b6278e07e736"
   },
   "source": [
    "Lets see first image in out test data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "_uuid": "bd3f2367733d37be74f26a8a53a8b33808eaf64b"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.imshow(X_test[0].reshape(96,96),cmap = 'gray')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_uuid": "c41bd777b9055ed782a475d88f9e7c0ac7d12a2b"
   },
   "source": [
    "Lets predict our results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "_uuid": "ecaf24956f805de32614c5476528102bdf56b329"
   },
   "outputs": [],
   "source": [
    "pred = model.predict(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_uuid": "a64f6787a940d21dfa976511f8880254bf14a8ab"
   },
   "source": [
    "Now the last step is the create our submission file keeping in the mind required format.\n",
    "There should be two columns :- RowId and Location\n",
    "Location column values should be filled according the lookup table provided ( IdLookupTable.csv)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "_uuid": "fff11927e693209f8f8d5029dffc74a53e1e8821"
   },
   "outputs": [],
   "source": [
    "lookid_list = list(lookid_data['FeatureName'])\n",
    "imageID = list(lookid_data['ImageId']-1)\n",
    "pre_list = list(pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "_uuid": "3d3c36d9375d4b81b4dd33cd1cc645f17b32023b"
   },
   "outputs": [],
   "source": [
    "rowid = lookid_data['RowId']\n",
    "rowid=list(rowid)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "_uuid": "de4123ef594738b0e23b485ba165f9ddc03b01de"
   },
   "outputs": [],
   "source": [
    "feature = []\n",
    "for f in list(lookid_data['FeatureName']):\n",
    "    feature.append(lookid_list.index(f))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "_uuid": "ae81c7411bedde81afbc315fbabaaf9f3223cff5"
   },
   "outputs": [],
   "source": [
    "preded = []\n",
    "for x,y in zip(imageID,feature):\n",
    "    preded.append(pre_list[x][y])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "_uuid": "a1949844f3012f5944a2159eeb8547d5b9434b4f"
   },
   "outputs": [],
   "source": [
    "rowid = pd.Series(rowid,name = 'RowId')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "_uuid": "bdc486586f354720bab48c2bbdb92287066cd977"
   },
   "outputs": [],
   "source": [
    "loc = pd.Series(preded,name = 'Location')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "_uuid": "fb144e2fa32cad9a8b334a7730a75bfed8c7ac0f"
   },
   "outputs": [],
   "source": [
    "submission = pd.concat([rowid,loc],axis = 1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "_uuid": "dc121456dbdb88e49605b255492262a77cc2ff75"
   },
   "outputs": [],
   "source": [
    "submission.to_csv('face_key_detection_submission.csv',index = False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
