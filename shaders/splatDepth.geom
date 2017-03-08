/*#version 330
	
layout(points) in;
layout(points, max_vertices=1) out;
	
uniform mat4 projMatrix;

in vec3 vNormal[];
in vec4 ellipse[];
in vec3 color[];
in float visible[];

out vec4 tPosinvA2invB2;
out vec3 vPos;
out vec3 normal;
out vec3 col;

void main()
{
	gl_Position = projMatrix * vec4( gl_in[0].gl_Position.xyz, 1 );;
	normal = vNormal[0];
	EmitVertex();
}*/

#version 330
	
layout(points) in;
layout(triangle_strip, max_vertices=4) out;
//layout(points, max_vertices=1) out;
	
uniform mat4 projMatrix;

in vec3 vNormal[];
in vec4 ellipse[];
in vec3 color[];
in float visible[];

out vec4 tPosinvA2invB2;
out vec3 vPos;
out vec3 normal;
out vec3 col;

void main()
{
	if (visible[0] == 1.0)
	{
		vec3 n = vNormal[0];
		
		vec3 s = ellipse[0].xyz;		// semimajor axis vec (side)
		vec3 t = cross( s, n );			// semiminor axis vec (top)
		
		float a = length( s );			// semimajor axis length
		float b = a;					// semiminor axis length (about to be scaled by proportion)	
		
		b *= ellipse[0].w;
		t *= ellipse[0].w;
						
		if (a > 0 && b > 0)
		{
			// 3D coordinates relative to splat center
			vec3 vP1 = + s - t;
			vec3 vP2 = - s - t; 
			vec3 vP3 = + s + t;
			vec3 vP4 = - s + t;

			// 2D axis aligned tangent coordinates in ellipse's plane
			mat3 R = transpose( mat3( s/a, t/b, n ) );
			vec2 t1 = (R * vP1).xy;
			vec2 t2 = (R * vP2).xy;
			vec2 t3 = (R * vP3).xy;
			vec2 t4 = (R * vP4).xy;
				
			// 3D view space coordinates
			vP1 += gl_in[0].gl_Position.xyz;
			vP2 += gl_in[0].gl_Position.xyz; 
			vP3 += gl_in[0].gl_Position.xyz;
			vP4 += gl_in[0].gl_Position.xyz;
				
			// clip space coordinates
			vec4 pP1 = projMatrix * vec4( vP1, 1 );
			vec4 pP2 = projMatrix * vec4( vP2, 1 ); 
			vec4 pP3 = projMatrix * vec4( vP3, 1 );
			vec4 pP4 = projMatrix * vec4( vP4, 1 );
				
			// diagonal of inverse characteristic matrix of the splat ellipse
			vec2 invA2invB2 = vec2(1 / (a*a), 1 / (b*b));
	
			normal = n;
			col = color[0];
				
			// TEMP:
			//id_ = id[0];
				
			gl_Position = pP1;
			tPosinvA2invB2 = vec4( t1, invA2invB2 );
			vPos = vP1;
			EmitVertex();

			gl_Position = pP2;
			tPosinvA2invB2 = vec4( t2, invA2invB2 );
			vPos = vP2;
			EmitVertex();

			gl_Position = pP3;
			tPosinvA2invB2 = vec4( t3, invA2invB2 );
			vPos = vP3;
			EmitVertex();

			gl_Position = pP4;
			tPosinvA2invB2 = vec4( t4, invA2invB2 );
			vPos = vP4;
			EmitVertex();
				
			EndPrimitive();
		}
	}	
}