/*
	CS 6023 Assignment 3. 
	Do not make any changes to the boiler plate code or the other files in the folder.
	Use cudaFree to deallocate any memory not in usage.
	Optimize as much as possible.
 */

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>

// calculating the initial transltions on the nodes given in translations vector
__global__ void calc_initial_node_tranlations(int T,int *d_trans, int* change_x, int* change_y){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < T){
    int node = d_trans[3 * id];
    int dir = d_trans[3 * id + 1];
    int how_much = d_trans[3 * id + 2];
    if(dir == 0) atomicSub(&change_x[node], how_much);
    else if(dir == 1) atomicAdd(&change_x[node], how_much);
    else if(dir == 2) atomicSub(&change_y[node], how_much);
    else if(dir == 3) atomicAdd(&change_y[node], how_much);
  }
}
// calculating the translations on the childeren of the nodes
__global__ void calc_adjacency_tranlations(int* d_hcsr, int num_edges, int s_idx, int* change_x, int* change_y, int node_id){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < num_edges){
    int node = d_hcsr[s_idx + id];
    change_x[node] = change_x[node] + change_x[node_id];
    change_y[node] = change_y[node] + change_y[node_id];
  }
}
// calculating resultant translations on all the nodes
__global__ void calc_resultant_tranlations(int* d_GlobalCoordinatesX, int* change_x, int* d_GlobalCoordinatesY, int* change_y, int V){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < V){
    d_GlobalCoordinatesX[id] = d_GlobalCoordinatesX[id] + change_x[id];
    d_GlobalCoordinatesY[id] = d_GlobalCoordinatesY[id] + change_y[id];
  }
}
// placing all the meshes on the frame
__global__ void place_mesh(int* d_GlobalCoordinatesX,int* d_GlobalCoordinatesY,int* mesh,int* d_FinalPng,int* d_opacity,int frameSizeX,int frameSizeY,int node_r,int node_c,int node_id,int opacity,int V){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int row = id / node_c;
  int col = id % node_c;
  bool isValid_r = row > -1 && row < node_r;
  bool isValid_c = col > -1 && col < node_c;
  if(isValid_r && isValid_c){
    int ROW = row + d_GlobalCoordinatesX[node_id];
    int COL = col + d_GlobalCoordinatesY[node_id];
    bool isValid_r = ROW > -1 && ROW < frameSizeX;
    bool isValid_c = COL > -1 && COL < frameSizeY;
    if (isValid_r && isValid_c){
      if(opacity >= d_opacity[ROW * frameSizeY + COL]){
        d_FinalPng[ROW * frameSizeY + COL] = mesh[row * node_c + col];
        d_opacity[ROW * frameSizeY + COL] = opacity;
      }
    }
  }
}
void readFile (const char *fileName, std::vector<SceneNode*> &scenes, std::vector<std::vector<int> > &edges, std::vector<std::vector<int> > &translations, int &frameSizeX, int &frameSizeY) {
	/* Function for parsing input file*/

	FILE *inputFile = NULL;
	// Read the file for input. 
	if ((inputFile = fopen (fileName, "r")) == NULL) {
		printf ("Failed at opening the file %s\n", fileName) ;
		return ;
	}

	// Input the header information.
	int numMeshes ;
	fscanf (inputFile, "%d", &numMeshes) ;
	fscanf (inputFile, "%d %d", &frameSizeX, &frameSizeY) ;
	

	// Input all meshes and store them inside a vector.
	int meshX, meshY ;
	int globalPositionX, globalPositionY; // top left corner of the matrix.
	int opacity ;
	int* currMesh ;
	for (int i=0; i<numMeshes; i++) {
		fscanf (inputFile, "%d %d", &meshX, &meshY) ;
		fscanf (inputFile, "%d %d", &globalPositionX, &globalPositionY) ;
		fscanf (inputFile, "%d", &opacity) ;
		currMesh = (int*) malloc (sizeof (int) * meshX * meshY) ;
		for (int j=0; j<meshX; j++) {
			for (int k=0; k<meshY; k++) {
				fscanf (inputFile, "%d", &currMesh[j*meshY+k]) ;
			}
		}
		//Create a Scene out of the mesh.
		SceneNode* scene = new SceneNode (i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity) ; 
		scenes.push_back (scene) ;
	}

	// Input all relations and store them in edges.
	int relations;
	fscanf (inputFile, "%d", &relations) ;
	int u, v ; 
	for (int i=0; i<relations; i++) {
		fscanf (inputFile, "%d %d", &u, &v) ;
		edges.push_back ({u,v}) ;
	}

	// Input all translations.
	int numTranslations ;
	fscanf (inputFile, "%d", &numTranslations) ;
	std::vector<int> command (3, 0) ;
	for (int i=0; i<numTranslations; i++) {
		fscanf (inputFile, "%d %d %d", &command[0], &command[1], &command[2]) ;
		translations.push_back (command) ;
	}
}


void writeFile (const char* outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY) {
	/* Function for writing the final png into a file.*/
	FILE *outputFile = NULL; 
	if ((outputFile = fopen (outputFileName, "w")) == NULL) {
		printf ("Failed while opening output file\n") ;
	}
	
	for (int i=0; i<frameSizeX; i++) {
		for (int j=0; j<frameSizeY; j++) {
			fprintf (outputFile, "%d ", hFinalPng[i*frameSizeY+j]) ;
		}
		fprintf (outputFile, "\n") ;
	}
}

int main (int argc, char **argv) {
	
	// Read the scenes into memory from File.
	const char *inputFileName = argv[1] ;
	int* hFinalPng ; 

	int frameSizeX, frameSizeY ;
	std::vector<SceneNode*> scenes ;
	std::vector<std::vector<int> > edges ;
	std::vector<std::vector<int> > translations ;
	readFile (inputFileName, scenes, edges, translations, frameSizeX, frameSizeY) ;
	hFinalPng = (int*) malloc (sizeof (int) * frameSizeX * frameSizeY) ;
	
	// Make the scene graph from the matrices.
    Renderer* scene = new Renderer(scenes, edges) ;

	// Basic information.
	int V = scenes.size () ;
	int E = edges.size () ;
	int numTranslations = translations.size () ;

	// Convert the scene graph into a csr.
	scene->make_csr () ; // Returns the Compressed Sparse Row representation for the graph.
	int *hOffset = scene->get_h_offset () ;  
	int *hCsr = scene->get_h_csr () ;
	int *hOpacity = scene->get_opacity () ; // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
	int **hMesh = scene->get_mesh_csr () ; // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
	int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX () ; // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
	int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY () ; // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
	int *hFrameSizeX = scene->getFrameSizeX () ; // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
	int *hFrameSizeY = scene->getFrameSizeY () ; // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

	auto start = std::chrono::high_resolution_clock::now () ;
	// Code begins here.
	// Do not change anything above this comment.

  // declaring queue for doing bfs
  std::queue<int> adjacent;
  int *d_FinalPng, *d_opacity;

  // performing required cudaMalloc
  cudaMalloc(&d_FinalPng, frameSizeX * frameSizeY * sizeof(int));
  cudaMalloc(&d_opacity, frameSizeX * frameSizeY * sizeof(int));
  // initialising opacities with INT_MIN
  cudaMemset(&d_opacity, frameSizeX * frameSizeY * sizeof(int), INT_MIN);

  int *d_GlobalCoordinatesX, *d_GlobalCoordinatesY;
  // performing required cudaMalloc and cudaMemcpy

  cudaMalloc(&d_GlobalCoordinatesX, V * sizeof(int));
	cudaMalloc(&d_GlobalCoordinatesY, V * sizeof(int));

	cudaMemcpy(d_GlobalCoordinatesX, hGlobalCoordinatesX, V * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_GlobalCoordinatesY, hGlobalCoordinatesY, V * sizeof(int), cudaMemcpyHostToDevice);

  int *d_translations, *h_trans = (int*)malloc(numTranslations * 3 * sizeof(int));
  cudaMalloc(&d_translations, numTranslations * 3 * sizeof(int));
  
  // creating a translations array from given vector
  for(int i=0; i<numTranslations; i++){
    h_trans[3 * i] = translations[i][0];
    h_trans[3 * i + 1] = translations[i][1];
    h_trans[3 * i + 2] = translations[i][2];
  }

  cudaMemcpy(d_translations, h_trans, numTranslations * 3 * sizeof(int), cudaMemcpyHostToDevice);

  int *d_hcsr;
  // performing required cudaMalloc and cudaMemcpy

	cudaMalloc(&d_hcsr, E * sizeof(int));
	cudaMemcpy(d_hcsr, hCsr, E * sizeof(int), cudaMemcpyHostToDevice);

  int *change_x, *change_y;
  cudaMalloc(&change_x, V * sizeof(int));
  cudaMalloc(&change_y,V * sizeof(int));

  cudaMemset(&change_x, V * sizeof(int), 0);
  cudaMemset(&change_y, V * sizeof(int), 0);


  int blk_size = 512;
  int num_blocks = (numTranslations + blk_size - 1) / blk_size;
  
  // calling kernel for calculating the initial transltions on the nodes given in translations vector
  calc_initial_node_tranlations<<<num_blocks, blk_size>>>(numTranslations, d_translations, change_x, change_y);

  cudaFree(d_translations);

  adjacent.push(0);
  
  while(!adjacent.empty()){
    int n_id = adjacent.front();
    adjacent.pop();
    if(hOffset[n_id] != hOffset[n_id + 1]){
      for(int adj = hOffset[n_id]; adj < hOffset[n_id + 1]; adj++){
        adjacent.push(hCsr[adj]);
      }
      blk_size = 512;
      num_blocks = ((hOffset[n_id + 1] - hOffset[n_id]) + blk_size - 1) / blk_size;
      
      // calling kernel for calculating the translations on the childeren of the nodes
      calc_adjacency_tranlations<<<num_blocks, blk_size>>>(d_hcsr, hOffset[n_id + 1] - hOffset[n_id], hOffset[n_id], change_x, change_y, n_id);
      cudaDeviceSynchronize();
    }
  }

  cudaFree(d_hcsr);

  blk_size = 512;
  num_blocks = (V + blk_size - 1) / blk_size;

  // calling kernel for calculating resultant translations on all the nodes
  calc_resultant_tranlations<<<num_blocks, blk_size>>>(d_GlobalCoordinatesX, change_x, d_GlobalCoordinatesY, change_y, V);
  cudaDeviceSynchronize();

  cudaFree(change_x);
  cudaFree(change_y);

  for(int m=0; m<V; m++){
    int r = hFrameSizeX[m]; 
    int c = hFrameSizeY[m];
    int o = hOpacity[m];
    int *mesh, *current_mesh = hMesh[m];
    cudaMalloc(&mesh, r * c * sizeof(int));
    cudaMemcpy(mesh, current_mesh, r * c * sizeof(int), cudaMemcpyHostToDevice);
    blk_size = 512;
    num_blocks = (r * c + blk_size - 1) / blk_size;

    // calling kernel to place m mesh on the frame
    place_mesh<<<num_blocks, blk_size>>>(d_GlobalCoordinatesX, d_GlobalCoordinatesY, mesh, d_FinalPng, d_opacity, frameSizeX, frameSizeY, r, c, m, o, V);
    cudaFree(mesh);
  }

  cudaFree(d_GlobalCoordinatesX);
  cudaFree(d_GlobalCoordinatesY);
  cudaFree(d_opacity);

  // copying results from gpu to cpu
  cudaMemcpy(hFinalPng, d_FinalPng, frameSizeX * frameSizeY * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_FinalPng);
	// Do not change anything below this comment.
	// Code ends here.

	auto end  = std::chrono::high_resolution_clock::now () ;

	std::chrono::duration<double, std::micro> timeTaken = end-start;

	printf ("execution time : %f\n", timeTaken) ;
	// Write output matrix to file.
	const char *outputFileName = argv[2] ;
	writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ;	

}
